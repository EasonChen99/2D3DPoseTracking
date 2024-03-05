import time
import random
import torch
import numpy as np
import pykitti
import mathutils
import visibility as visibility
import sys
sys.path.append('core')
from utils_point import rotate_back, mat2xyzrpy, overlay_imgs, invert_pose, to_rotation_matrix
from data_preprocess import Data_preprocess
from camera_model import CameraModel
from scipy.optimize import least_squares
import cv2

class LSGC:
    def __init__(self, flow, local_map, point2d_cur, point2d_next,
                 calib, real_shape,
                 occlusion_threshold, occlusion_kernel,
                 velo2cam2, device, x, y):
        self.flow = flow
        self.local_map = local_map.to(device)
        self.point2d_cur = point2d_cur.transpose()
        self.point2d_next = point2d_next.transpose()
        self.calib = calib
        self.real_shape = real_shape
        self.occlusion_threshold = occlusion_threshold
        self.occlusion_kernel = occlusion_kernel
        self.velo2cam2 = velo2cam2
        self.device = device
        self.x = x
        self.y = y

        self.data_generate = Data_preprocess(self.calib, self.occlusion_threshold, self.occlusion_kernel)
        self.data_generate.real_shape = self.real_shape
        self.Pc_vel2cam = torch.tensor([[0, -1, 0, 0], [0, 0, -1, 0],
                                   [1, 0, 0, 0], [0, 0, 0, 1]],
                                  dtype=self.local_map.dtype, device=self.device)
        self.Pc_cam2vel = self.Pc_vel2cam.inverse()


    def depth_generation(self, pc, only_uv=False):
        cam_params = self.calib
        cam_model = CameraModel()
        cam_model.focal_length = cam_params[:2]
        cam_model.principal_point = cam_params[2:]
        uv, depth, _, refl, indexes = cam_model.project_withindex_pytorch(pc, self.real_shape)
        uv_noindex = cam_model.project_withoutindex_pytorch(pc)
        uv = uv.t().int().contiguous()
        uv_noindex = uv_noindex.t().int().contiguous()
        if only_uv:
            return uv, uv_noindex, indexes
        depth_img = torch.zeros(self.real_shape[:2], device=self.device, dtype=torch.float)
        depth_img += 1000.
        idx_img = (-1) * torch.ones(self.real_shape[:2], device=self.device, dtype=torch.float)
        indexes = indexes.float()
        depth_img, idx_img = visibility.depth_image(uv, depth, indexes,
                                                    depth_img, idx_img,
                                                    uv.shape[0], self.real_shape[1], self.real_shape[0])
        depth_img[depth_img == 1000.] = 0.

        deoccl_index_img = (-1) * torch.ones(self.real_shape[:2], device=self.device, dtype=torch.float)
        projected_points = torch.zeros_like(depth_img, device=self.device)
        projected_points, _ = visibility.visibility2(depth_img, cam_params,
                                                     idx_img,
                                                     projected_points,
                                                     deoccl_index_img,
                                                     depth_img.shape[1],
                                                     depth_img.shape[0],
                                                     self.occlusion_threshold,
                                                     int(self.occlusion_kernel))
        projected_points /= 100.
        projected_points = projected_points.unsqueeze(0)

        return projected_points


    def xyzrpy2mat(self, parameters):
        cameras_parameter = parameters.reshape((2, 6))

        cur_r = cameras_parameter[0, :3]
        cur_t = cameras_parameter[0, 3:6]
        cur_r = mathutils.Quaternion(cur_r).to_matrix()
        cur_r.resize_4x4()
        cur_t = mathutils.Matrix.Translation(cur_t)
        RT_cur = cur_t * cur_r
        RT_cur = torch.tensor(RT_cur, dtype=torch.float).to(self.device)
        next_r = cameras_parameter[1, :3]
        next_t = cameras_parameter[1, 3:6]
        next_r = mathutils.Quaternion(next_r).to_matrix()
        next_r.resize_4x4()
        next_t = mathutils.Matrix.Translation(next_t)
        RT_next = next_r * next_t
        RT_next = torch.tensor(RT_next, dtype=torch.float).to(self.device)

        return RT_cur, RT_next


    def makeEquFLow(self, RT_pred, RT_pred_next, points_3d):
        """
        Make the equal optical flow between the current frame and the next frame based on the predicted posed.

        Args:
            local_map(torch.Tensor): Point cloud to be transformed, shape [4xN]
            RT_pred(torch.Tensor): Predicted camera pose of the current frame, shape [4x4]
            RT_pred_next(torch.Tensor): Predicted camera pose of the next frame, shape [4x4]
        """
        ## generate current reprojection
        RT_pred = torch.mm(torch.mm(self.velo2cam2, RT_pred.inverse()), self.velo2cam2.inverse())
        cur_local_map = rotate_back(torch.mm(self.Pc_vel2cam, points_3d), RT_pred)
        cur_local_map = torch.mm(self.Pc_cam2vel, cur_local_map)
        cur_uv, cur_uv_noindex, cur_indexes = self.depth_generation(cur_local_map, only_uv=True)

        ## generate next reprojection
        RT_pred_next = torch.mm(torch.mm(self.velo2cam2, RT_pred_next.inverse()), self.velo2cam2.inverse())
        next_local_map = rotate_back(torch.mm(self.Pc_vel2cam, points_3d), RT_pred_next)
        next_local_map = torch.mm(self.Pc_cam2vel, next_local_map)
        next_uv, next_uv_noindex, next_indexes = self.depth_generation(next_local_map, only_uv=True)

        ## generate equal flow between two reprojections
        delta, indexes = self.data_generate.delta_1(cur_uv, next_uv, cur_indexes, next_indexes)
        indexes_cur = cur_indexes[indexes]
        indexes_cur = torch.arange(indexes_cur.shape[0]).to(self.device) + 1
        indexes_next = next_indexes[indexes]
        indexes_next = torch.arange(indexes_next.shape[0]).to(self.device) + 1
        cur_uv_af_index = cur_uv[indexes[cur_indexes], :]
        mask1 = indexes_cur > 0
        mask2 = indexes_next > 0
        mask = mask1 & mask2

        delta_img = self.data_generate.delta_2(delta, cur_uv_af_index, mask)
        delta_img = delta_img[:, self.x:self.x + 320, self.y:self.y + 960]

        return delta_img, cur_uv_noindex.cpu().detach().numpy(), next_uv_noindex.cpu().detach().numpy()


    def func(self, parameters, flag=0):
        RT_cur, RT_next = self.xyzrpy2mat(parameters)

        equal_flow, cur_uv, next_uv = self.makeEquFLow(RT_cur, RT_next, self.local_map)

        # flow error
        mask = (equal_flow[0, ...] != 0) + (equal_flow[1, ...] != 0)
        residues_flow = (self.flow - equal_flow) * mask
        residues_flow = residues_flow.cpu().detach().numpy() / (mask.sum().cpu().detach().numpy() + 1e-5)
        residues_flow = residues_flow.ravel()

        if flag == 1:
            return residues_flow

        # reprojection error
        error_cur = np.linalg.norm(self.point2d_cur - cur_uv, axis=1) / (cur_uv.shape[0] + 1e-5)
        error_next = np.linalg.norm(self.point2d_next - next_uv, axis=1) / (next_uv.shape[0] + 1e-5)
        residues_reproj = np.concatenate((error_cur.ravel(), error_next.ravel()))

        return np.concatenate((residues_flow, 1e-3 * residues_reproj))

    def visualizationRT(self, RT, img, img_name):
        RT = torch.mm(torch.mm(self.velo2cam2, RT.inverse()), self.velo2cam2.inverse())
        pc_rotated = rotate_back(torch.mm(self.Pc_vel2cam, self.local_map), RT)
        pc_rotated = torch.mm(self.Pc_cam2vel, pc_rotated)
        depth = self.depth_generation(pc_rotated)
        depth = depth[:, self.x:self.x + 320, self.y:self.y + 960]
        depth = depth.unsqueeze(0)
        out0 = overlay_imgs(img, depth[0, 0, :, :])
        cv2.imwrite(img_name, out0)

        return out0

    def run(self, RT_pred, RT_pred_next, count=0):
        ## initial camera parameter
        cur_rt = mat2xyzrpy(RT_pred)
        cur_cameras = torch.cat((cur_rt[3:], cur_rt[:3]), 0)
        next_rt = mat2xyzrpy(RT_pred_next)
        next_cameras = torch.cat((next_rt[3:], next_rt[:3]), 0)

        x0 = torch.hstack((cur_cameras, next_cameras)).cpu().numpy()
        res = least_squares(self.func, x0,
                            verbose=0, x_scale='jac',
                            ftol=1e-8, xtol=1e-8, gtol=1e-8,
                            method='lm', max_nfev=500)

        if count <= 99 and np.sum(self.func(res.x, flag=1)) == 0:
            max_angle = 0.01
            max_t = 0.01
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-max_t, max_t)
            transl_y = np.random.uniform(-max_t, max_t)
            transl_z = np.random.uniform(-max_t, max_t)
            R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
            T = mathutils.Vector((transl_x, transl_y, transl_z))
            R, T = invert_pose(R, T)
            R, T = torch.tensor(R), torch.tensor(T)
            RT_random = to_rotation_matrix(R, T)
            RT_cur, RT_next = self.run(RT_pred, torch.mm(RT_random.to(RT_pred.device), RT_pred), count=count+1)
            return RT_cur, RT_next

        RT_cur, RT_next = self.xyzrpy2mat(res.x)

        return RT_cur, RT_next

