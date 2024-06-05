import csv
import open3d as o3
import os
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import visibility as visibility
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
from PIL import Image
from tqdm import tqdm
import pykitti
import pandas as pd
import argparse
import cv2
import sys

from core.camera_model import CameraModel
from core.raft import RAFT
from core.utils_point import quaternion_from_matrix, to_rotation_matrix, overlay_imgs
from core.utils import count_parameters, recover_rgb
from core.flow2pose import Flow2Pose
from core.depth_completion import sparse_to_dense
from core.quaternion_distances import quaternion_distance
from core.data_preprocess import Data_preprocess
from core.GC_optimal import LSGC
from core.utils_vo import VO_pose
from core.flow_viz import flow_to_image

from RAFT.core.raft_f2f import RAFT_f2f


def get_calib_kitti(sequence):
    if sequence in [0, 1, 2]:
        return torch.tensor([718.856, 718.856, 607.1928, 185.2157])
    elif sequence == 3:
        return torch.tensor([721.5377, 721.5377, 609.5593, 172.854])
    elif sequence in [4, 5, 6, 7, 8, 9, 10]:
        return torch.tensor([707.0912, 707.0912, 601.8873, 183.1104])
    else:
        raise TypeError("Sequence Not Available")

def load_map(map_file, device):
    downpcd = o3.io.read_point_cloud(map_file)
    voxelized = torch.tensor(downpcd.points, dtype=torch.float)
    voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float)), 1)
    voxelized = voxelized.t()
    voxelized = voxelized.to(device)
    return voxelized

def load_GT_poses(dataset_dir, seq_id='00'):
    all_files = []
    GTs_R = []
    GTs_T = []
    df_locations = pd.read_csv(os.path.join(dataset_dir, seq_id, 'poses.csv'), sep=',', dtype={'timestamp': str})
    for index, row in df_locations.iterrows():
        if not os.path.exists(f"{dataset_dir}/{seq_id}/image_2/{int(row['timestamp']):06d}.png"):
            continue
        all_files.append(f"{int(row['timestamp']):06d}")
        GT_T = torch.tensor([float(row['x']), float(row['y']), float(row['z'])])
        GT_R = torch.tensor([float(row['qw']), float(row['qx']), float(row['qy']), float(row['qz'])])
        GTs_R.append(GT_R)
        GTs_T.append(GT_T)

    return all_files, GTs_R, GTs_T

def crop_local_map(PC_map, pose, velo2cam):
    local_map = PC_map.clone()
    pose = pose.inverse()
    local_map = torch.mm(pose, local_map).t()
    indexes = local_map[:, 1] > -25.
    indexes = indexes & (local_map[:, 1] < 25.)
    indexes = indexes & (local_map[:, 0] > -10.)
    indexes = indexes & (local_map[:, 0] < 100.)
    local_map = local_map[indexes]

    local_map = torch.mm(velo2cam, local_map.t())

    local_map = local_map[[2, 0, 1, 3], :]

    return local_map

def depth_generation(local_map, image_size, cam_params, occu_thre, occu_kernel, device, only_uv=False):
    cam_model = CameraModel()
    cam_model.focal_length = cam_params[:2]
    cam_model.principal_point = cam_params[2:]
    uv, depth, _, refl, indexes = cam_model.project_withindex_pytorch(local_map, image_size)
    uv = uv.t().int().contiguous()
    if only_uv:
        return uv, indexes
    depth_img = torch.zeros(image_size[:2], device=device, dtype=torch.float)
    depth_img += 1000.
    idx_img = (-1) * torch.ones(image_size[:2], device=device, dtype=torch.float)
    indexes = indexes.float()

    depth_img, idx_img = visibility.depth_image(uv, depth, indexes,
                                                depth_img, idx_img,
                                                uv.shape[0], image_size[1], image_size[0])
    depth_img[depth_img == 1000.] = 0.

    deoccl_index_img = (-1) * torch.ones(image_size[:2], device=device, dtype=torch.float)
    projected_points = torch.zeros_like(depth_img, device=device)
    projected_points, _ = visibility.visibility2(depth_img, cam_params,
                                                 idx_img,
                                                 projected_points,
                                                 deoccl_index_img,
                                                 depth_img.shape[1],
                                                 depth_img.shape[0],
                                                 occu_thre,
                                                 int(occu_kernel))
    projected_points /= 100.
    projected_points = projected_points.unsqueeze(0)

    return projected_points

def custom_transform(rgb):
    to_tensor = transforms.ToTensor()
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    rgb = to_tensor(rgb)
    rgb = normalization(rgb)
    return rgb


class I2D_VO(torch.nn.Module):
    def __init__(self, args):
        super(I2D_VO, self).__init__()
        self.args = args
        self.image2depth_net_a = RAFT(args)
        if args.tight_couple:
            self.image2depth_net_b = RAFT(args)
            self.image2image_net = RAFT_f2f(args)

    def forward(self, cur_frame, next_frame, dense_depth, sparse_depth, iters):

        _, I2Dflow_up_cur = self.image2depth_net_a(dense_depth, cur_frame,
                                                   lidar_mask=sparse_depth,
                                                   iters=iters, test_mode=True)
        if self.args.tight_couple:
            _, I2Dflow_up_next = self.image2depth_net_a(dense_depth, next_frame,
                                                        lidar_mask=sparse_depth,
                                                        iters=iters, test_mode=True)

            cur_frame = recover_rgb(cur_frame)
            next_frame = recover_rgb(next_frame)
            _, I2Iflow_up = self.image2image_net(cur_frame, next_frame,
                                                 iters=iters, test_mode=True)

            return I2Dflow_up_cur, I2Dflow_up_next, I2Iflow_up

        return I2Dflow_up_cur


def main(args):
    print(args)
    # set device
    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.set_device(args.gpus[0])

    # initialize camera parameters
    calib = get_calib_kitti(args.test_sequence)
    calib = calib.to(device)

    # load LiDAR map
    if args.maps_file is not None:
        maps_file = args.maps_file
    else:
        print("LiDAR map does not exist")
        sys.exit()
    print(f'load pointclouds from {maps_file}')
    vox_map = load_map(maps_file, device)
    print(f'load pointclouds finished! {vox_map.shape[1]} points')

    # initialize kitti parameters
    test_sequence = f"{args.test_sequence:02d}"
    kitti_folder = os.path.split(args.data_folder[:-1])[0]
    kitti = pykitti.odometry(kitti_folder, test_sequence)
    velo2cam2 = kitti.calib.T_cam2_velo
    velo2cam2 = torch.from_numpy(velo2cam2).float().to(device)

    # load GT poses
    print('load ground truth poses')
    all_files, GTs_R, GTs_T = load_GT_poses(args.data_folder, test_sequence)
    print(len(all_files))

    # load model parameters
    model = torch.nn.DataParallel(I2D_VO(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))
    model.load_state_dict(torch.load(args.load_checkpoints, map_location='cuda:0'))
    model.to(device)
    model.eval()

    # initialize filter parameters
    occlusion_threshold = args.occlusion_threshold
    occlusion_kernel = args.occlusion_kernel

    # open log
    if args.save_log:
        log_file = f'./logs/Ours_KITTI04_0.csv'
        log_file_f = open(log_file, 'w')
        log_file = csv.writer(log_file_f)
        header = [f'timestamp', f'x', f'y', f'z',
                  f'qx', f'qy', f'qz', f'qw']
        log_file.writerow(header)


    est_rot = []
    est_trans = []
    est_rot.append(GTs_R[0].to(device))
    est_trans.append(GTs_T[0].to(device))
    err_t_list = []
    err_r_list = []
    # outliers = []
    print('Start tracking using I2D-Loc...')
    k = 0
    end = time.time()
    for idx in range(len(all_files)-1):
        idx = idx + k
        if idx == k:
            initial_R = GTs_R[idx].to(device)
            initial_T = GTs_T[idx].to(device)
            # impose random error
            max_t = 0.
            max_angle = 0.
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-max_t, max_t)
            transl_y = np.random.uniform(-max_t, max_t)
            transl_z = np.random.uniform(-max_t, min(max_t, 1.))
            import mathutils
            from core.utils_point import invert_pose
            R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
            T = mathutils.Vector((transl_x, transl_y, transl_z))
            R, T = invert_pose(R, T)
            R, T = torch.tensor(R), torch.tensor(T)
            RT_err = to_rotation_matrix(R, T)
        else:
            # pose initialization
            initial_R = est_rot[idx-k]
            initial_T = est_trans[idx-k]

        RT = to_rotation_matrix(initial_R, initial_T)
        if idx == k:
            RT = torch.mm(RT, RT_err.to(device))
        RT = RT.to(device)

        cur = Image.open(os.path.join(args.data_folder, test_sequence, 'image_2', all_files[idx] + '.png'))
        cur = custom_transform(cur)
        next = Image.open(os.path.join(args.data_folder, test_sequence, 'image_2', all_files[idx + 1] + '.png'))
        next = custom_transform(next)
        real_shape = [cur.shape[1], cur.shape[2], cur.shape[0]]

        local_map = crop_local_map(vox_map, RT, velo2cam2)
        projected_points = depth_generation(local_map, real_shape, calib, occlusion_threshold, occlusion_kernel, device)

        # crop image and depth
        x = (cur.shape[1] - 320) // 2
        y = (cur.shape[2] - 960) // 2
        cur = cur[:, x:x + 320, y:y + 960]
        cur = cur.unsqueeze(0)
        next = next[:, x:x + 320, y:y + 960]
        next = next.unsqueeze(0)
        sparse = projected_points[:, x:x + 320, y:y + 960]
        sparse = sparse.unsqueeze(0)

        # dilation
        dense = []
        for i in range(sparse.shape[0]):
            depth_img = sparse[i, 0, :, :].cpu().numpy() * 100.
            depth_img_dilate = sparse_to_dense(depth_img.astype(np.float32))
            dense.append(depth_img_dilate / 100.)
        dense = torch.tensor(dense).float().to(device)
        dense = dense.unsqueeze(1)

        # flag = True
        flag = False
        if flag:
            original_overlay = overlay_imgs(cur[0, :, :, :], sparse[0, 0, :, :].clone())
            cv2.imwrite(f"./visualization/original/{idx:05d}_img.png", original_overlay)
            original_overlay = overlay_imgs(cur[0, :, :, :], sparse[0, 0, :, :].clone())
            cv2.imwrite(f"./visualization/original/{idx:05d}_depth.png", original_overlay)
            sys.exit()

        # run model
        if args.tight_couple:
            flow_up_cur, flow_up_next, flow_up = model(cur, next, dense, sparse, iters=args.iters)
            # print(flow_up_cur.shape, flow_up_next.shape, flow_up.shape)
        else:
            flow_up_cur = model(cur, next, dense, sparse, iters=args.iters)

        # update current pose
        R_pred, T_pred, depth_img_cur, pc_project_uv_cur = \
                            Flow2Pose(flow_up_cur, sparse, calib, return_medium=True)
        R_pred[2] *= -1
        R_pred[3] *= -1
        T_pred[1] *= -1
        T_pred[2] *= -1
        RT_pred = to_rotation_matrix(R_pred, T_pred)
        RT_pred = RT_pred.to(device)

        # loose couple
        if args.loose_couple:
            alpha = 10  # 10
            beta = 200  # 200
            # alpha = -1  # 10
            # beta = 1000000000  # 200

            RT_new_flow = torch.mm(RT, RT_pred)
            predicted_T_flow = RT_new_flow[:3, 3]
            err_t_flow = torch.norm(predicted_T_flow.to(device) - est_trans[idx - k].to(device)) * 100.

            if err_t_flow < alpha or err_t_flow > beta:
                ## VO
                cur = recover_rgb(cur)
                cur = cur[0, ...].permute(1, 2, 0).cpu().numpy()
                cur = cv2.convertScaleAbs(cur)
                next = recover_rgb(next)
                next = next[0, ...].permute(1, 2, 0).cpu().numpy()
                next = cv2.convertScaleAbs(next)
                Rt_vo = VO_pose(cur, next, calib.cpu().numpy())
                RT_vo = torch.tensor(Rt_vo, dtype=RT.dtype).to(device)
                RT_new = torch.mm(RT, RT_vo)
                print(f"{idx}" + "*" * 80)
            else:
                RT_new = RT_new_flow
        elif not args.tight_couple:
            RT_new = torch.mm(RT, RT_pred)

        # tight couple
        if args.tight_couple:
            R_pred_next, T_pred_next, depth_img_next, pc_project_uv_next = \
                                Flow2Pose(flow_up_next, sparse, calib, return_medium=True)
            R_pred_next[2] *= -1
            R_pred_next[3] *= -1
            T_pred_next[1] *= -1
            T_pred_next[2] *= -1
            RT_pred_next = to_rotation_matrix(R_pred_next, T_pred_next)
            RT_pred_next = RT_pred_next.to(device)

            ######################################### optimization algorithm #########################################
            cam_model = CameraModel()
            cam_params = calib.cpu().numpy()
            x, y = 28, 140
            cam_params[2] = cam_params[2] + 480 - (y + y + 960) / 2.
            cam_params[3] = cam_params[3] + 160 - (x + x + 320) / 2.
            cam_model.focal_length = cam_params[:2]
            cam_model.principal_point = cam_params[2:]

            # reconstruct 3d point clouds
            depth_img_mask = (depth_img_cur > 0) * (depth_img_next > 0)
            depth_img_common = np.zeros(depth_img_cur.shape, dtype=depth_img_cur.dtype)
            depth_img_common[depth_img_mask] = depth_img_cur[depth_img_mask]
            point3d_common = cam_model.depth2pc(depth_img_common) # LiDAP_map * pose^-1

            # generate corresponding 2d points
            pc_project_uv_cur_u = pc_project_uv_cur[:, :, 0][depth_img_mask]
            pc_project_uv_cur_v = pc_project_uv_cur[:, :, 1][depth_img_mask]
            point2d_cur = np.array([pc_project_uv_cur_v, pc_project_uv_cur_u])
            pc_project_uv_next_u = pc_project_uv_next[:, :, 0][depth_img_mask]
            pc_project_uv_next_v = pc_project_uv_next[:, :, 1][depth_img_mask]
            point2d_next = np.array([pc_project_uv_next_v, pc_project_uv_next_u])

            LSGC_solver = LSGC(flow_up[0, ...],
                               point3d_common, point2d_cur, point2d_next,
                               calib, real_shape,
                               occlusion_threshold, occlusion_kernel,
                               velo2cam2, device, x, y)
            opti_RT_cur, opti_RT_next = LSGC_solver.run(RT_pred, RT_pred_next)

            RT_new = torch.mm(RT, opti_RT_cur)
            RT_new_next = torch.mm(RT, opti_RT_next)

            predicted_R_next = quaternion_from_matrix(RT_new_next)
            predicted_T_next = RT_new_next[:3, 3]

        # calculate RTE RRE
        predicted_R = quaternion_from_matrix(RT_new)
        predicted_T = RT_new[:3, 3]
        err_r = quaternion_distance(predicted_R.unsqueeze(0).to(device),
                                    GTs_R[idx].unsqueeze(0).to(device), device=device)
        err_r = err_r * 180. / math.pi
        err_t = torch.norm(predicted_T.to(device) - GTs_T[idx].to(device)) * 100.
        err_r_list.append(err_r.item())
        err_t_list.append(err_t.item())

        # print(f"{idx:05d}: {np.mean(err_t_list):.5f} {np.mean(err_r_list):.5f} {np.median(err_t_list):.5f} "
        #       f"{np.median(err_r_list):.5f} {(time.time()-end)/(idx+1):.5f}")
        print(f"{idx:05d}: {np.mean(err_t_list):.5f} {np.mean(err_r_list):.5f} {np.std(err_t_list):.5f} "
              f"{np.std(err_r_list):.5f} {(time.time()-end)/(idx+1):.5f}")

        # update pose list
        est_rot[idx-k] = predicted_R
        est_trans[idx-k] = predicted_T
        if not args.tight_couple:
            est_rot.append(predicted_R)
            est_trans.append(predicted_T)
        else:
            est_rot.append(predicted_R_next)
            est_trans.append(predicted_T_next)

        if args.save_log:
            predicted_T = predicted_T.cpu().numpy()
            predicted_R = predicted_R.cpu().numpy()

            log_string = [all_files[idx], str(predicted_T[0]), str(predicted_T[1]), str(predicted_T[2]),
                          str(predicted_R[1]), str(predicted_R[2]), str(predicted_R[3]), str(predicted_R[0])]
            log_file.writerow(log_string)

        if args.render:
            original_overlay = overlay_imgs(cur[0, :, :, :], sparse[0, 0, :, :])
            cv2.imwrite(f"./visualization/cur/{idx:05d}.png", original_overlay)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, metavar='DIR',
                        default='/media/eason/e835c718-d773-44a1-9ca4-881204d9b53d/Datasets/KITTI/sequences',
                        help='path to dataset')
    parser.add_argument('--test_sequence', type=int, default=0)
    parser.add_argument('--occlusion_kernel', type=float, default=5.)
    parser.add_argument('--occlusion_threshold', type=float, default=3.)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--use_reflectance', action='store_true')
    parser.add_argument('-cps', '--load_checkpoints', type=str)
    parser.add_argument('--loose_couple', action='store_true')
    parser.add_argument('--tight_couple', action='store_true')
    parser.add_argument('--maps_file', type=str, default='/media/eason/e835c718-d773-44a1-9ca4-881204d9b53d/Datasets/KITTI/sequences/00/map-00.pcd')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    main(args)
