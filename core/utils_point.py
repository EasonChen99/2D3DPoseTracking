import math
import cv2
from PIL import Image
import mathutils
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from matplotlib import cm
from torch.utils.data.dataloader import default_collate


def rotate_points(PC, R, T=None, inverse=True):
    if T is not None:
        R = R.to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(T)
        RT = T*R
    else:
        RT=R.copy()
    if inverse:
        RT.invert_safe()
    RT = torch.tensor(RT, device=PC.device, dtype=torch.float)

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC


def rotate_points_torch(PC, R, T=None, inverse=True):
    if T is not None:
        R = quat2mat(R)
        T = tvector2mat(T)
        RT = torch.mm(T, R)
    else:
        RT = R.clone()
    if inverse:
        RT = RT.inverse()
    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC


def rotate_forward(PC, R, T=None):
    """
    Transform the point cloud PC, so to have the points 'as seen from' the new
    pose T*R
    Args:
        PC (torch.Tensor): Point Cloud to be transformed, shape [4xN] or [Nx4]
        R (torch.Tensor/mathutils.Euler): can be either:
            * (mathutils.Euler) euler angles of the rotation part, in this case T cannot be None
            * (torch.Tensor shape [4]) quaternion representation of the rotation part, in this case T cannot be None
            * (mathutils.Matrix shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
            * (torch.Tensor shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
        T (torch.Tensor/mathutils.Vector): Translation of the new pose, shape [3], or None (depending on R)

    Returns:
        torch.Tensor: Transformed Point Cloud 'as seen from' pose T*R
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC, R, T, inverse=True)
    else:
        return rotate_points(PC, R, T, inverse=True)


def rotate_back(PC_ROTATED, R, T=None):
    """
    Inverse of :func:`~utils.rotate_forward`.
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC_ROTATED, R, T, inverse=False)
    else:
        return rotate_points(PC_ROTATED, R, T, inverse=False)


def invert_pose(R, T):
    """
    Given the 'sampled pose' (aka H_init), we want CMRNet to predict inv(H_init).
    inv(T*R) will be used as ground truth for the network.
    Args:
        R (mathutils.Euler): Rotation of 'sampled pose'
        T (mathutils.Vector): Translation of 'sampled pose'

    Returns:
        (R_GT, T_GT) = (mathutils.Quaternion, mathutils.Vector)
    """
    R = R.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(T)
    RT = T * R
    RT.invert_safe()
    T_GT, R_GT, _ = RT.decompose()
    return R_GT.normalized(), T_GT


def merge_inputs(queries):
    point_clouds = []
    imgs = []
    reflectances = []
    returns = {key: default_collate([d[key] for d in queries]) for key in queries[0]
               if key != 'point_cloud' and key != 'rgb' and key != 'reflectance'}
    for input in queries:
        point_clouds.append(input['point_cloud'])
        imgs.append(input['rgb'])
        if 'reflectance' in input:
            reflectances.append(input['reflectance'])
    returns['point_cloud'] = point_clouds
    returns['rgb'] = imgs
    if len(reflectances) > 0:
        returns['reflectance'] = reflectances
    return returns


def quaternion_from_matrix(matrix):
    """
    Convert a rotation matrix to quaternion.
    Args:
        matrix (torch.Tensor): [4x4] transformation matrix or [3,3] rotation matrix.

    Returns:
        torch.Tensor: shape [4], normalized quaternion
    """
    if matrix.shape == (4, 4):
        R = matrix[:-1, :-1]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = torch.zeros(4, device=matrix.device)
    if tr > 0.:
        S = (tr+1.0).sqrt() * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]).sqrt() * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]).sqrt() * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]).sqrt() * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / q.norm()


def quatmultiply(q, r):
    """
    Multiply two quaternions
    Args:
        q (torch.Tensor/nd.ndarray): shape=[4], first quaternion
        r (torch.Tensor/nd.ndarray): shape=[4], second quaternion

    Returns:
        torch.Tensor: shape=[4], normalized quaternion q*r
    """
    t = torch.zeros(4, device=q.device)
    t[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    t[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    t[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    t[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
    return t / t.norm()


def quat2mat(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [4x4] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((4, 4), device=q.device)
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    mat[3, 3] = 1.
    return mat


def tvector2mat(t):
    """
    Translation vector to homogeneous transformation matrix with identity rotation
    Args:
        t (torch.Tensor): shape=[3], translation vector

    Returns:
        torch.Tensor: [4x4] homogeneous transformation matrix

    """
    assert t.shape == torch.Size([3]), "Not a valid translation"
    mat = torch.eye(4, device=t.device)
    mat[0, 3] = t[0]
    mat[1, 3] = t[1]
    mat[2, 3] = t[2]
    return mat


def mat2xyzrpy(rotmatrix):
    """
    Decompose transformation matrix into components
    Args:
        rotmatrix (torch.Tensor/np.ndarray): [4x4] transformation matrix

    Returns:
        torch.Tensor: shape=[6], contains xyzrpy
    """
    roll = math.atan2(-rotmatrix[1, 2], rotmatrix[2, 2])
    pitch = math.asin ( rotmatrix[0, 2])
    yaw = math.atan2(-rotmatrix[0, 1], rotmatrix[0, 0])
    x = rotmatrix[:3, 3][0]
    y = rotmatrix[:3, 3][1]
    z = rotmatrix[:3, 3][2]

    return torch.tensor([x, y, z, roll, pitch, yaw], device=rotmatrix.device, dtype=rotmatrix.dtype)

def xyzrpy2mat(xyzrpy):
    """
    Compose transformation matrix from components
    Args:
        xyzrpy (torch.Tensor/np.ndarray): [6] xyzrpy

    Returns:
        torch.Tensor: [4x4] transformation matrix
    """
    x, y, z, roll, pitch, yaw = xyzrpy.tolist()

    # Define rotation matrix elements
    c_roll = math.cos(roll)
    s_roll = math.sin(roll)
    c_pitch = math.cos(pitch)
    s_pitch = math.sin(pitch)
    c_yaw = math.cos(yaw)
    s_yaw = math.sin(yaw)

    # Construct rotation matrix
    R_x = torch.tensor([[1, 0, 0],
                        [0, c_roll, -s_roll],
                        [0, s_roll, c_roll]])

    R_y = torch.tensor([[c_pitch, 0, s_pitch],
                        [0, 1, 0],
                        [-s_pitch, 0, c_pitch]])

    R_z = torch.tensor([[c_yaw, -s_yaw, 0],
                        [s_yaw, c_yaw, 0],
                        [0, 0, 1]])

    # Combine rotation matrices
    R = torch.mm(torch.mm(R_x, R_y), R_z)

    # Construct transformation matrix
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = torch.tensor([x, y, z])

    return T


def to_rotation_matrix(R, T):
    R = quat2mat(R)
    T = tvector2mat(T)
    RT = torch.mm(T, R)
    return RT

def rotation_vector_to_euler(rvecs):
    R, _ = cv2.Rodrigues(rvecs)

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def overlay_imgs(rgb, lidar):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    rgb = rgb.clone().cpu().permute(1,2,0).numpy()
    rgb = rgb*std+mean

    # rgb = cv2.resize(rgb, (120, 40))
    # lidar = lidar.cpu().numpy()
    # lidar = cv2.resize(lidar, (120, 40))

    lidar[lidar == 0] = 1000.
    lidar = -lidar

    lidar = lidar.clone()
    lidar = lidar.unsqueeze(0)
    lidar = lidar.unsqueeze(0)

    lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = -lidar
    lidar[lidar == 1000.] = 0.

    lidar = lidar[0][0]


    lidar = lidar.cpu().numpy()
    min_d = 0
    max_d = np.max(lidar)
    lidar = ((lidar - min_d) / (max_d - min_d)) * 255
    lidar = lidar.astype(np.uint8)
    lidar_color = cm.jet(lidar)
    lidar_color[:, :, 3] = 0.5
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) \
                  + rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)

    blended_img = cv2.cvtColor((blended_img*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    # cv2.imwrite(f'./images/output/{idx:06d}_{name}.png', blended_img)
    return blended_img


def mat2euler(M, cy_thresh=None, seq='zyx'):
    '''
    Taken From: http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
    Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
     threshold below which to give up on straightforward arctan for
     estimating x rotation.  If None (default), estimate from
     precision of input.
    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
     Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
    [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
    [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
    [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
     z = atan2(-r12, r11)
     y = asin(r13)
     x = atan2(-r23, r33)
    for x,y,z order
    y = asin(-r31)
    x = atan2(r32, r33)
    z = atan2(r21, r11)
    Problems arise when cos(y) is close to zero, because both of::
     z = atan2(cos(y)*sin(z), cos(y)*cos(z))
     x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33 * r33 + r23 * r23)
    if seq == 'zyx':
        if cy > cy_thresh:  # cos(y) not close to zero, standard form
            z = math.atan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13, cy)  # atan2(sin(y), cy)
            x = math.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21, r22)
            y = math.atan2(r13, cy)  # atan2(sin(y), cy)
            x = 0.0
    elif seq == 'xyz':
        if cy > cy_thresh:
            y = math.atan2(-r31, cy)
            x = math.atan2(r32, r33)
            z = math.atan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi / 2
                x = atan2(r12, r13)
            else:
                y = -np.pi / 2
    else:
        raise Exception('Sequence not recognized')
    return z, y, x

import functools

def euler2mat(z=0, y=0, x=0, isRadian=True):
    ''' Return matrix for rotations around z, y and x axes
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    M : array shape (3,3)
         Rotation matrix giving same rotation as for given angles
    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True
    The output rotation matrix is equal to the composition of the
    individual rotations
    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True
    You can specify rotations by named arguments
    >>> np.all(M3 == euler2mat(x=xrot))
    True
    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.
    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)
    Rotations are counter-clockwise.
    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True
    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''

    if not isRadian:
        z = ((np.pi) / 180.) * z
        y = ((np.pi) / 180.) * y
        x = ((np.pi) / 180.) * x
    assert z >= (-np.pi) and z < np.pi, 'Inapprorpriate z: %f' % z
    assert y >= (-np.pi) and y < np.pi, 'Inapprorpriate y: %f' % y
    assert x >= (-np.pi) and x < np.pi, 'Inapprorpriate x: %f' % x

    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
            [[cosy, 0, siny],
             [0, 1, 0],
             [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
            [[1, 0, 0],
             [0, cosx, -sinx],
             [0, sinx, cosx]]))
    if Ms:
        return functools.reduce(np.dot, Ms[::-1])
    return np.eye(3)

