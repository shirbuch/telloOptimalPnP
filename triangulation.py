#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""See https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html for matlab-numpy translation tips"""
from typing import Tuple, List, SupportsInt, Union

import numpy as np
import cv2 as cv
import camera_params


def triangulateOnePoint(point1, point2, P1, P2):
    A = np.zeros((4, 4))

    maz = [P1[i][-1] for i in range(len(P1))]
    maz2 = [P2[i][-1] for i in range(len(P2))]

    A[:2] = np.array(point1).reshape((2, 1)) * maz - P1.T[0:2]
    A[2:] = np.array(point2).reshape((2, 1)) * maz2 - P2.T[0:2]

    # alternatively:
    # def do_thing(point, P):
    #         return point*P[2] - P[:3]
    # A = np.stack((do_thing(point1, P1), do_thing(point2, P2)))

    _, _, V = np.linalg.svd(A)
    V = np.array(V).T
    X = V[:, -1]
    return X[:3] / X[-1]  # de_hom


def translation(points1, points2):
    #    points = [[430.7418212890625, 162.26895141601562], [272.48809814453125, 422.48956298828125], [402.5331115722656, 226.53968811035156], [304.48907470703125, 392.49652099609375]]

    points = np.array(points1).reshape(len(points), 2)
    points2 = np.array(points2).reshape(len(points), 2)

    K1_trans = np.array(camera_params.K1).transpose()
    K2_trans = np.array(camera_params.K2).transpose()

    I3 = np.eye(3)
    T0 = np.zeros((1, 3))

    T = np.array(camera_params.T).reshape(1, 3)

    P1 = np.dot(np.concatenate((I3, T0), axis=0), K1_trans)
    P2 = np.dot(np.concatenate((camera_params.R, T), axis=0), K2_trans)
    result = [triangulateOnePoint(np.array(p1), np.array(p2), P1, P2) for p1, p2 in zip(points, points2)]

    return result


def PnP(real_points: Union[List[List[int]], np.ndarray],
        screen_points: Union[List[List[int]], np.ndarray],
        index: int
        ) -> Tuple[np.ndarray, np.ndarray]:
    inner_calibration = [camera_params.TELLO, camera_params.DistCoff]

    # ret, rv, tv = cv.solvePnP(np.array(real_points), np.array(screen_points), np.array(inner_calibration[index - 1]), camera_params.DistCoff[index - 1])
    # ret, rv, tv, _ = cv.solvePnPRansac(cv.UMat(np.array(real_points, dtype=np.float32)), cv.UMat(np.array(screen_points, dtype=np.float32)),
    #                                    np.array(inner_calibration[index - 1]), camera_params.DistCoff[index - 1])
    ret, rv, tv = cv.solvePnP(np.array(real_points, dtype=np.float32),
                              np.array(screen_points, dtype=np.float32),
                              np.array(inner_calibration[0]),
                              inner_calibration[1])
    return rv, tv


def GetCamera4x4ProjMat(rvec, tvec):
    """
    Retrieves 4x4 projection matrix in homogenius coordinates
    """
    res = cv.Rodrigues(rvec)[0]
    temp = np.hstack((res, tvec))
    return temp


def GetCamera3x4ProjMat(rvec, tvec, K):
    """
    Retrieves 3x4 projection from data SolvePNP returns
    """
    res = cv.Rodrigues(rvec)[0]
    temp = np.hstack((res, tvec))
    return np.dot(K, temp)


def Get3DFrom4D(p4d):
    """
    Returns 3d coordinate from the homogenius coord
    """
    p4d /= p4d[3]
    p4d = p4d[0:3]
    return p4d.T


def get_2d_matrix(points):
    """
    returns a 2xN matrix of the points

    :param points: List of 2d points
    :return: 2xN cv matrix
    """
    x = np.array([pt.point[0] for pt in points])
    y = np.array([pt.point[1] for pt in points])
    return np.vstack((x, y))


def find_real_points(R1, T1, R2, T2, m1, m2):
    """
    calculates the 3D location of the new points

    :param R1: Rotation of camera 1
    :param T1: Translation of camera 1
    :param R2: Rotation of camera 2
    :param T2: Translation of camera 2
    :param m1: points from camera 1
    :param m2: matching points from camera 2
    """
    if len(m1) == 0 or len(m2) == 0 or len(m1) != len(m2):  # no new points
        return

    proj1 = GetCamera3x4ProjMat(R1, T1, camera_params.K1)
    proj2 = GetCamera3x4ProjMat(R2, T2, camera_params.K2)
    print('---------------------')
    print(proj1)
    print(proj2)
    print('---------------------')
    p3d = cv.triangulatePoints(proj1, proj2, get_2d_matrix(m1), get_2d_matrix(m2))
    p3d = Get3DFrom4D(p3d)
    # p3d = cv.convertPointsFromHomogeneous(p3d)
    for (real, p1, p2) in zip(p3d, m1, m2):
        # if real[2] < 3 and real[2] > -3:
        p1.realLoc = p2.realLoc = real


def get_camera_position(RT):
    """
    returns the camera position

    :param RT: tuple of R and T
    :return: 3x3 Matrix
    """
    if RT is None:
        return None
    rotation = cv.Rodrigues(RT[0])[0]
    return -np.dot(np.transpose(rotation), RT[1])
