import numpy as np
import torch
import sys
sys.path.append("core")
from utils_point import rotate_forward, rotate_back

class CameraModel:
    def __init__(self, focal_length=None, principal_point=None):
        self.focal_length = focal_length
        self.principal_point = principal_point

    def project_pytorch(self, xyz: torch.Tensor, image_size, reflectance=None):
        if xyz.shape[0] == 3:
            xyz = torch.cat([xyz, torch.ones(1, xyz.shape[1], device=xyz.device)])
        else:
            if not torch.all(xyz[3, :] == 1.):
                xyz[3, :] = 1.
                raise TypeError("Wrong Coordinates")
        order = [1, 2, 0, 3]
        xyzw = xyz[order, :]
        indexes = xyzw[2, :] >= 0
        if reflectance is not None:
            reflectance = reflectance[:, indexes]
        xyzw = xyzw[:, indexes]

        uv = torch.zeros((2, xyzw.shape[1]), device=xyzw.device)
        uv[0, :] = self.focal_length[0] * xyzw[0, :] / xyzw[2, :] + self.principal_point[0]
        uv[1, :] = self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]
        indexes = uv[0, :] >= 0.1
        indexes = indexes & (uv[1, :] >= 0.1)
        indexes = indexes & (uv[0,:] < image_size[1])
        indexes = indexes & (uv[1,:] < image_size[0])
        if reflectance is None:
            uv = uv[:, indexes], xyzw[2, indexes], xyzw[:3, indexes], None
        else:
            uv = uv[:, indexes], xyzw[2, indexes], xyzw[:3, indexes], reflectance[:, indexes]

        return uv

    # for pc_RT
    def project_withindex_pytorch(self, xyz: torch.Tensor, image_size, reflectance=None):
        if xyz.shape[0] == 3:
            xyz = torch.cat([xyz, torch.ones(1, xyz.shape[1], device=xyz.device)])
        else:
            if not torch.all(xyz[3, :] == 1.):
                xyz[3, :] = 1.
                # raise TypeError("Wrong Coordinates")
        order = [1, 2, 0, 3]
        xyzw = xyz[order, :]
        indexes = xyzw[2, :] >= 0
        if reflectance is not None:
            reflectance = reflectance[:, indexes]
        xyzw = xyzw[:, indexes]

        VI_indexes = indexes

        uv = torch.zeros((2, xyzw.shape[1]), device=xyzw.device)
        uv[0, :] = self.focal_length[0] * xyzw[0, :] / xyzw[2, :] + self.principal_point[0]
        uv[1, :] = self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]
        indexes = uv[0, :] >= 0.1
        indexes = indexes & (uv[1, :] >= 0.1)
        indexes = indexes & (uv[0, :] < image_size[1])
        indexes = indexes & (uv[1, :] < image_size[0])

        # generate complete indexes
        ind = torch.where(VI_indexes == True)[0]
        VI_indexes[ind] = VI_indexes[ind] & indexes


        if reflectance is None:
            uv = uv[:, indexes], xyzw[2, indexes], xyzw[:3, indexes], None, VI_indexes
        else:
            uv = uv[:, indexes], xyzw[2, indexes], xyzw[:3, indexes], reflectance[:, indexes], VI_indexes

        return uv

    def project_withoutindex_pytorch(self, xyz: torch.Tensor):
        """not consider potential outer point"""
        if xyz.shape[0] == 3:
            xyz = torch.cat([xyz, torch.ones(1, xyz.shape[1], device=xyz.device)])
        else:
            if not torch.all(xyz[3, :] == 1.):
                xyz[3, :] = 1.
                raise TypeError("Wrong Coordinates")
        order = [1, 2, 0, 3]
        xyzw = xyz[order, :]

        uv = torch.zeros((2, xyzw.shape[1]), device=xyzw.device)
        uv[0, :] = self.focal_length[0] * xyzw[0, :] / xyzw[2, :] + self.principal_point[0]
        uv[1, :] = self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]

        return uv

    def get_matrix(self):
        matrix = np.zeros([3, 3])
        matrix[0, 0] = self.focal_length[0]
        matrix[1, 1] = self.focal_length[1]
        matrix[0, 2] = self.principal_point[0]
        matrix[1, 2] = self.principal_point[1]
        matrix[2, 2] = 1.0
        return matrix

    def deproject_pytorch(self, uv, pc_project_uv):
        index = np.argwhere(uv > 0)
        mask = uv > 0
        z = uv[mask]
        x = (index[:, 1] - self.principal_point[0]) * z / self.focal_length[0]
        y = (index[:, 0] - self.principal_point[1]) * z / self.focal_length[1]
        zxy = np.array([z, x, y])
        zxy = torch.tensor(zxy, dtype=torch.float32)
        zxyw = torch.cat([zxy, torch.ones(1, zxy.shape[1], device=zxy.device)])
        zxy = zxyw[:3, :]
        zxy = zxy.cpu().numpy()
        xyz = zxy[[1, 2, 0], :]


        pc_project_u = pc_project_uv[:, :, 0][mask]
        pc_project_v = pc_project_uv[:, :, 1][mask]
        pc_project = np.array([pc_project_v, pc_project_u])


        match_index = np.array([index[:, 0], index[:, 1]])

        return xyz.transpose(), pc_project.transpose(), match_index.transpose()

    def depth2pc(self, uv):
        index = np.argwhere(uv > 0)
        mask = uv > 0
        z = uv[mask]
        x = (index[:, 1] - self.principal_point[0]) * z / self.focal_length[0]
        y = (index[:, 0] - self.principal_point[1]) * z / self.focal_length[1]

        zxy = np.array([z, x, y], dtype=np.float32)
        zxy = torch.tensor(zxy, dtype=torch.float32)
        zxyw = torch.cat([zxy, torch.ones(1, zxy.shape[1], device=zxy.device)])

        return zxyw