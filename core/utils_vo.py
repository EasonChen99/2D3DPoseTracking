import cv2
import numpy as np
import sys
from utils_point import mat2euler, euler2mat


def VO_pose(prev_img, curr_img, calib, x, y):
    orb = cv2.SIFT_create()

    kp1, des1 = orb.detectAndCompute(prev_img, None)
    kp2, des2 = orb.detectAndCompute(curr_img, None)

    # use FLANN matcher
    indexParams = dict(algorithm=0, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)
    good_match = []
    for j, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good_match.append(m)
    matches = good_match

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # When the dataset is KITTI
    calib[2] = calib[2] + 480 - (y + y + 960) / 2.
    calib[3] = calib[3] + 160 - (x + x + 320) / 2.

    K = np.array([[calib[0], 0., calib[2]],
                  [0., calib[1], calib[3]],
                  [0., 0., 1.]])

    E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K)

    # get camera motion
    R = R.transpose()
    t = -np.matmul(R, t)

    R = mat2euler(R)
    R = euler2mat(-R[1], -R[2], R[0])
    t = t[[2, 0, 1]]
    t[1] *= -1
    t[2] *= -1

    R_vo = np.eye(4)
    R_vo[:3, :3] = R
    t_vo = np.eye(4)
    t_vo[:3, 3] = t[:, 0]
    RT_vo = np.matmul(t_vo, R_vo)
    return RT_vo