#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2 as cv

FEATURES_TO_TRACK = 4000
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
LIMIT = 1.64  # good according to my checks + to the Flenn paper
DEFAULT_SAVE_MODE = "csv"
MERGE_DIST = 15  # Threshold
SAME_DESC = 550  # Threshold
SAME_DESC1 = 300  # Threshold
HAMMING = 40  # Threshold
SAME_DISTANCE = 5  # Threshold
MATCHING_RATE = 50

#
# MERGE_DIST = 15  # Threshold
# SAME_DESC = 550  # Threshold
# SAME_DESC1 = 300  # Threshold
# HAMMING = 40  # Threshold
# SAME_DISTANCE = 5  # Threshold
MIN_FOLLOW_POINTS = 100
MAX_FOLLOW_POINTS = 10
MIN_MATCHING = 4
BOARD_SIZE = (9, 6)
SQURESIZE = 1.4
CIRCLE_SIZE = 4
KNN_MATCHES = 2
BEST_FEATURES = 1

# GFFT_feature_params = dict(
#     maxCorners=FEATURES_TO_TRACK,
#     qualityLevel=0.02,
#     minDistance=10,
#     blockSize=3
# )
# OPTIC_FLOW_lk_params = dict(
#     winSize=(50, 50),
#     maxLevel=100,
#     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
# )

GFFT_feature_params = dict(
    maxCorners = FEATURES_TO_TRACK,
    qualityLevel = 0.2,
    minDistance = 10,
    blockSize = 3
)

# Parameters for lucas kanade optical flow
OPTIC_FLOW_lk_params = dict(
    winSize =(50,50),
    maxLevel=100,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)

# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.03)
