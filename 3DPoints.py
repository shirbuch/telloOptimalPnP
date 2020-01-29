#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import time
from typing import List, Tuple, Text, AnyStr, Union, Optional, Iterator

import cv2 as cv
import numpy as np
import copy
import pickle
import math

import triangulation
import camera_params
import constants

# features_detectores_mode = {
#     "GFTT" = find_features_GFTT()
#     "SIFT" = find_features_SIFT()
#     "ORB " = find_features_ORB()
# }

sift = cv.xfeatures2d.SIFT_create(constants.FEATURES_TO_TRACK)

orb = cv.ORB_create(
    nfeatures=constants.FEATURES_TO_TRACK,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20
)

SAVE_VIDEO = False  # True to save the video, False to not save
REAL_TIME = True  # True to use cameras in real time, false to use video offline

video_name = "New_Stairs"
vid_file1 = "outputs/" + video_name + "1.avi"
vid_file2 = "outputs/" + video_name + "2.avi"
output_name1 = "outputs/" + video_name + "1_points" + ".avi"
output_name2 = "outputs/" + video_name + "2_points" + ".avi"

matching_vid = "outputs/" + video_name + "_matching" + ".avi"

if REAL_TIME:
    vid_file1 = 1
    vid_file2 = 2

saved_points = []
curr_num = 0
min_featured = constants.MIN_FOLLOW_POINTS
cams_update = False


def convert_to_tuple_list(arr: Optional[Union[List, List[List]]] = None,
                          idx: Optional[int] = None
                          ) -> List[Tuple]:
    """
    turns  a list of lists into a list of tuples

    :param arr: list of lists
    :param idx: index of an element in each list

    :return: list of tuples; if the index exists, returns a list of the indexes of all the lists
    """
    if arr is None:
        return [tuple()]
    if idx is not None:
        arr = (item[idx] for item in arr)  # We can use generator here. Performance boost.

    return [tuple(item) for item in arr]


class Tracking_Point():
    """

    Attributes:
    :param name: string of 3 english letters that represent a name
    :param point: tuple that represent a point in the plain
    :param keyPoint: keyPoint of that point
    :param descriptor: desocriptor of the keypoint
    :param realLoc: real world location
    :param matchedPoint: the matched point in the second camera
    """

    def __init__(self, name, point, keyPoint, descriptor, realLoc=None):
        self.name = name
        self.point = tuple(point[0] if len(point) == 1 else point)
        self.keyPoint = keyPoint
        self.descriptor = descriptor
        self.realLoc = realLoc
        self.matchedPoint = None
        self.isOld = False


def find_features_ORB(frame):
    """
    finds the featured points using the ORB algorithm

    :param frame: a single frame from the video

    :return: list of the featured points
    :return: list of keyPoints
    :return: SIFT descriptor
    """
    kp, desc = orb.detectAndCompute(frame, None)
    points = [k.pt for k in kp]
    return points, kp, desc


def find_features_GFTT(frame, max_features=constants.FEATURES_TO_TRACK):
    """
    finds the featured points using the GFTT algorithm

    :param frame: a single frame from the video
    :param max_features: the maximal number of features the algorithm will return

    :return: list of the featured points
    :return: list of keyPoints
    :return: SIFT descriptor
    """
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Shift frame to grayScale

    constants.GFFT_feature_params['maxCorners'] = max_features
    points = cv.goodFeaturesToTrack(
        frame,
        mask=None,
        **constants.GFFT_feature_params
    )  # get the features from opencv

    points = convert_to_tuple_list(points.tolist(), idx=0)  # Normalize to array of tuples

    kp = points_to_keyPoints(points, constants.GFFT_feature_params["blockSize"])  # get list of keyPoints
    desc = get_sift_desc(frame, kp)  # get descriptor
    return points, kp, desc


def find_features_SIFT(frame):
    """
    finds the featured points using the SIFT algorithm

    :param frame: a single frame from the video

    :return: list of the featured points
    :return: list of keyPoints
    :return: SIFT descriptor
    """
    kp, desc = sift.detectAndCompute(frame, None)
    # kp = kp[::3] # Ignore the appearance of the same point 3 times
    points = [k.pt for k in kp]
    return points, kp, desc


def points_to_keyPoints(points, size=10):
    """
    returns keyPoints of points

    :param points: list of points as tuples/lists
    :param size: points diameter

    :return: list of keyPoints
    """
    return [cv.KeyPoint(point[0], point[1], size) for point in points]


def get_sift_desc(frame, kpoint):
    """
    gets the descriptors of the points with sift

    :param frame: a single frame from the video
    :param kpoint: list of keyPoints

    :return: descriptor
    """
    return sift.compute(frame, kpoint)[1]


def get_orb_desc(frame, kpoint):
    """
    gets the descriptors of the points with orb

    :param frame: a single frame from the video
    :param kpoint: list of keyPoints

    :return: descriptor
    """
    return orb.compute(frame, kpoint)[1]


def opticFlow(old_frame: np.ndarray,
              frame: np.ndarray,
              points: List[np.ndarray]
              ) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """
    updates the points from the previous frame to the new one

    :param old_frame: the previous frame
    :param frame: the new frame
    :param points: points on screen

    :return new_points: the new updated points
    :return st: a binary list - 1 if the point 'i' is in the frame, 0 otherwise
    """

    # points = [[np.array(point)] for point in points]
    points = np.array(points).astype('float32')

    new_points: np.ndarray
    st = err = None
    new_points, st, err = cv.calcOpticalFlowPyrLK(
        old_frame,
        frame,
        points,
        None,
        **constants.OPTIC_FLOW_lk_params
    )

    #new_points = new_points.reshape([-1, 1, 2])  # -1 is any number, 2 is for x,y # TODO: Reomove this 1
    #new_points = convert_to_tuple_list(new_points.tolist(), idx=0)  # Normalize to array of tuples
    return new_points, st


index_params = dict(algorithm=constants.FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params, search_params)  # TODO: Understand

# Euclidean distance
def distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1.point[1]) - np.array(pt2.point[1]))


# TODO(tomer or Fares): fix that one point can be mapped to more than one point
def match_points(points: List[Tracking_Point],
                 point_set: List[Tracking_Point],
                 DESC_LIMIT: Optional[int] = 70
                 ) -> Union[bool,
                            Tuple[List[Tracking_Point], List[Tracking_Point], List[Tracking_Point], List[List[int]]]]:
    """
    Takes two sets of points and find a "quick" matching between them

    :arg points: The point set we would like to compare
    :arg point_set: The set to whom we compare to
    :arg DESC_LIMIT: (Optional, default: 100, too high to catch anything) The max distance between paired points

    :returns match_set1:    Tracking_Point list
    :returns match_set2:    Tracking_Point list paired to match_set1
    :returns matches:       The list we get from the Flann algorithm
    :returns matchesMask:   Flags out the good pairs
    """

    desc1 = np.float32(np.array([pt.descriptor for pt in points]))  # list of descriptors of the input points
    desc2 = np.float32(np.array([pt.descriptor for pt in point_set]))  # list of descriptors of the saved set of points

    # matches = flann.knnMatch(np.asarray(desc1, np.float32), np.asarray(desc2, np.float32), k=constants.KNN_MATCHES)

    matches = flann.knnMatch(desc1, desc2, k=constants.KNN_MATCHES)  # finds the k best matches for each descriptor desc1
    matchesMask = [[0, 0]] * len(matches)

    goodMatches = []  # list of the "good  matching" points
    for i, (m, n) in enumerate(matches):
        if m.distance < constants.LIMIT * n.distance:  # checks if the match is within the distance ratio, if so its a good pair
            matchesMask[i] = [1, 0]  # flag the good pair
            if m.distance < DESC_LIMIT:  # if it is in the limit of the max distance we'll add it to goodMatches
                # if distance(points[m.queryIdx], point_set[m.trainIdx]) < constants.SAME_DISTANCE:
                goodMatches.append(m)

    if len(goodMatches) == 0:  # if goodMatches is empty - no matches
        print("ERROR: NO MATCHES")
        return [], [], [], []

    match_set1 = [points[x.queryIdx] for x in goodMatches]  # put the good points in 'points' in order #TODO check
    match_set2 = [point_set[x.trainIdx] for x in goodMatches]  # put the good points in 'point_set' in order
    for pt1, pt2 in zip(match_set1, match_set2):
        pt1.matchedPoint = pt2
        pt2.matchedPoint = pt1
        # if np.array_equal(pt1.realLoc, None):  #TODO: Check! - not none
        #     pt1.realLoc = pt2.realLoc
        # if np.array_equal(pt2.realLoc, None):
        #     pt2.realLoc = pt1.realLoc

    return match_set1, match_set2, matches, matchesMask


def save_points(Tracking_Points: List[Tracking_Point],
                mode: Optional[Text] = constants.DEFAULT_SAVE_MODE,
                basename: Optional[AnyStr] = "model"
                ) -> None:
    """
    Save `Tracking_Points` to a file.

    :arg Tracking_Points: The points to save.
    :arg mode: (Optional, default: `constants.DEFAULT_SAVE_MODE`)
        Valid values:
        * "csv": Appends to the csv file
        * "pickle": Overwrite the previuos pickle file
    :arg basename: (Optional, default: "model") The basename for the file.

    :returns: `None`.
    """
    global saved_points
    if mode == "csv":  # saves in csv format
        with open(f"{basename}.csv", "wt") as f:
            # Add unvisited points to file
            for tp in Tracking_Points:
                # f.write("{name},{pt},{kp},{desc},{x},{y},{z}\n".format(
                #     name=tp.name,
                #     pt=tp.point,
                #     kp=tp.keyPoint,
                #     desc=tp.descriptor,
                #     x=tp.realLoc[0],
                #     y=tp.realLoc[1],
                #     z=tp.realLoc[2]
                # )
                f.write("{x},{y},{z}\n".format(
                    x=tp.realLoc[0],
                    y=tp.realLoc[1],
                    z=tp.realLoc[2]
                )
                )
                #saved_points.append(tp)
    elif mode == "pickle":  # saves in pickle format
        with open(f"{basename}.pickle", "w") as f:
            pickle.dump(Tracking_Point, f)
            saved_points += Tracking_Points
    # TODO: Add XML option
    else:
        print("Unknown saving format \"{0}\". Not saving the points...".format(mode))


# TODO maybe change one day
def next_name() -> Text:
    """
    returns the next available prefix to name the points

    :return: string
    """
    global curr_num
    curr_num += 1
    return (
            str(chr(ord('A') + curr_num // 26 ** 2))
            + str(chr(ord('A') + (curr_num // 26) % 26))
            + str(chr(ord('A') + (curr_num % 26)))
    )


def merge_same_point_by_pos(my_points_temp: List[Tracking_Point],
                            on_screen: List[Tracking_Point]
                            ) -> Union[List[Tracking_Point], Iterator[Tracking_Point]]:
    """
    Remove duplicate points using their positions (if they are too close to each other)

    :arg my_points_temp: Points we would like to check if already appears.
    :arg on_screen: The points that on screen by now.

    :return my_points_temp: After removing points that we have in `on_screen` already.
    :return merge_on_screen: Points that are on screen but we already know
    """
    merge_on_screen = set()

    def filterFunc(point):
        """
        decides if a point should be merged or not

        :param point: a point

        :return: false if the point is too close to a point in 'my_points_temp', otherwise true
        """
        for scPt in on_screen:  # for each point on screen
            if point is not scPt and np.linalg.norm(np.array(point.point) - np.array(
                    scPt.point)) < constants.MERGE_DIST:  # if 'scPt' isn't 'point' and its distance to each other is less than MERGE_DIST
                print(point.name + " removed DIST by " + scPt.name)
                # scPt.point = point.point
                merge_on_screen.add(scPt)
                return False
        return True

    my_points_temp = filter(filterFunc,
                            my_points_temp)  # puts each point in the list only if 'filterFunc' returs true, aka isn't very close to another point
    return my_points_temp, merge_on_screen


# TODO(olicht): fix to use match point now
def merge_same_point_by_desc(my_points_temp: List[Tracking_Point],
                             my_points: List[Tracking_Point]
                             ) -> Union[List[Tracking_Point], Iterator[Tracking_Point]]:
    """
    Remove duplicate points using their descriptors (if their descriptors are too close to each other)

    :arg my_points_temp: Points we would like to check if already appears.
    :arg my_points: All of the points.

    :return my_points_temp: After removing points that we have in `my_points` already.
    :return merge_on_screen: Points that are on screen but we already know
    """
    merge_on_screen = set()

    def filterFunc(point):
        """
        decides if a point should be merged or not

        :param point: a point

        :return: false if the point is too close to a point on screen, otherwise true
        """
        for mp in my_points:  # for each point in all of the points
            if point is not mp and np.linalg.norm(
                    point.descriptor - mp.descriptor) < constants.SAME_DESC:  # if 'mp' isn't 'point' and its distance to their descriptors is less than SAME_DESC
                print(point.name + " removed DESC by " + mp.name)
                mp.point = point.point
                merge_on_screen.add(mp)  # add the points that are on screen
                mp.isOld = True
                return False
        return True

    my_points_temp = filter(filterFunc,
                            my_points_temp)  # puts each point in the list only if 'filterFunc' returns true, aka isn't already on screen
    return my_points_temp, merge_on_screen


def merge_same_point(my_points_temp: List[Tracking_Point],
                     my_points: List[Tracking_Point],
                     on_screen: List[Tracking_Point]
                     ) -> Tuple[List[Tracking_Point], List[Tracking_Point]]:
    """
    merge the points that are the same to us (close enough to each other)

    :param my_points_temp: Points we would like to check if already appears (from the current frame, on screen)
    :param my_points: All of the points
    :param on_screen: The points that on screen by now

    :return my_points: list of all of the points after merging
    :return on_screen: list of the points that on screen by now after merging
    """
    my_points_temp, merge_on_screen1 = merge_same_point_by_pos(my_points_temp, on_screen)  # merge by distance
    my_points_temp, merge_on_screen2 = merge_same_point_by_desc(my_points_temp, my_points)  # merge by descriptors
    my_points_temp = list(my_points_temp)
    on_screen_temp = list(merge_on_screen1.union(merge_on_screen2))
    on_screen_temp = [item for item in on_screen_temp if item not in on_screen]
    for pt in my_points_temp:
        pt.name = next_name()

    my_points += my_points_temp  # updates 'my_points'
    on_screen = my_points_temp + on_screen_temp + [item for item in on_screen if
                                                   item not in on_screen_temp]  # updates 'on_screen' - 'my_points_temp': new on screen, 'on_screen_temp': old on screen

    return my_points, on_screen


# to move to better place in code
def record_cams():
    """
    Record video from the two cameras and save it
    """
    cap1 = cap2 = out1 = out2 = None  # For code-style check
    try:
        cap1 = cv.VideoCapture(1)  # capture camera 1
        cap2 = cv.VideoCapture(2)  # capture camera 2

        WIDTH = int(cap1.get(3))  # width of video
        HEIGHT = int(cap1.get(4))  # height of video

        size = (WIDTH, HEIGHT)  # dimension of video

        fourcc = cv.VideoWriter_fourcc(*'MPEG')  # 'x264' doesn't work

        out1 = cv.VideoWriter(vid_file1, fourcc, 20.0, size)  # get video 1
        out2 = cv.VideoWriter(vid_file2, fourcc, 20.0, size)  # get video 2

        while cap1.isOpened() and cap2.isOpened():
            _, frame1 = cap1.read()  # read frame from cam1
            _, frame2 = cap2.read()  # read frame from cam2

            out1.write(frame1)  # write frame1
            out2.write(frame2)  # write frame2

            frame1 = cv.transpose(frame1)
            frame2 = cv.transpose(frame2)

            frame1 = cv.flip(frame1, 1)
            frame2 = cv.flip(frame2, 0)

            cv.imshow('frame1', frame1)  # show frame1
            cv.imshow('frame2', frame2)  # show frame2

            k = cv.waitKey(1) & 0xFF
            if k == ord('q') or k == ord('Q') or k == 27:  # 'Q', 'q', or Esc in order to stop showing the frames
                break
    finally:
        cap1 and cap1.release()
        cap2 and cap2.release()

        out1 and out1.release()
        out2 and out2.release()

        cv.destroyAllWindows()


# TODO(olicht): maybe optimize
def manage_points(frame: np.ndarray,
                  old_frame: np.ndarray,
                  my_points: List[Tracking_Point],
                  on_screen: List[Tracking_Point],
                  ) -> Tuple[List[Tracking_Point], List[Tracking_Point]]:
    """
    Finds new points if needed (amount of on screen too low).
    Removes duplicates and updates both lists.
    If no need to find new points, it follows the old points and update their location on screen.

    :arg frame: The current frame
    :arg old_frame: The previous frame
    :arg my_points: All the points this camera document
    :arg on_screen: All the points that appear now on the screen

    :return my_points: Now includes the new points from the current frame
    :return on_screen: All points discovered now, plus the ones was on screen before
    """
    global min_featured
    global cams_update
    st = None
    try:
        if old_frame is not None:  # if it isn't the first frame
            points = [mp.point for mp in on_screen]
            new_points, st = opticFlow(old_frame, frame, points)  # get the updated points and the status
            # if sum(st) >= min_featured:  # if there are enough points on screen
            new_on_screen = []
            new_keyPoints = points_to_keyPoints(new_points)

            for mp, np, kp, s in zip(on_screen, new_points, new_keyPoints, st):  # updates the points
                mp.point = np
                mp.keyPoint = kp
                if s:  # if the point is on the screen, add it to 'new_on_screen'
                    new_on_screen.append(mp)
            on_screen = new_on_screen
    except cv.error as e:  # fixed?
        if str(e).find("1244") > 0:
            st = [0]
            print("Avoided not enough points")
        else:
            raise

    if old_frame is None or sum(st) < min_featured or cams_update:  # if it is the first frame or there aren't enough points on screen
        ps, kps, desc = find_features_GFTT(frame, constants.FEATURES_TO_TRACK - sum(st if st is not None else [0]))  # ps - featured points, kps - the matching keyPoints, d - the matching descriptors
        # because we don't have enough points on screen (or non if its the first frame) we need to find new featured points (can be different from the ones we have now)
        # ps, kps, desc = find_features_SIFT(frame)

        if old_frame is None:  # first frame
            my_points_temp = [Tracking_Point(next_name(), p, k, d, None) for p, k, d in zip(ps, kps,
                                                                                                    desc)]  # create a 'Tracking_Point' variable to ach featured point, puts all in a list
        else:
            my_points_temp = [Tracking_Point("Temp", p, k, d, None) for p, k, d in zip(ps, kps,
                                                                                               desc)]  # create a 'Tracking_Point' variable to ach featured point, puts all in a list

        if old_frame is None:  # if it is the first frame
            my_points = my_points_temp[:]  # copy
            on_screen = my_points_temp[:]  # copy
        else:  # if not, we need to check for duplicates
            my_points, on_screen = merge_same_point(my_points_temp, my_points, on_screen)

        # min_featured = min(max(len(on_screen), constants.MAX_FOLLOW_POINTS),constants.MIN_FOLLOW_POINTS)  # if after adding all the points there isn't enough, we will lower the threshold in order to stabilize
        min_featured = constants.MIN_FOLLOW_POINTS

        cams_update = not cams_update

        print("---------------------")
        print("New Points")
        print("---------------------")

    return my_points, on_screen


# TODO: check if good
def both_cam_points(points1: List[Tracking_Point],
                    points2: List[Tracking_Point]
                    ) -> Tuple[List[Tracking_Point], List[Tracking_Point]]:

    for pt1 in points1:
        for pt2 in points2:
            if np.linalg.norm(pt1.descriptor - pt2.descriptor) <= constants.SAME_DESC:
                # pt1.matchedPoint = pt2
                # pt2.matchedPoint = pt1
                pt2.name = pt1.name
    # both_points1 = []
    # both_points2 = []
    # for point in points2:
    #     dists = ((mp, np.linalg.norm(point.descriptor - mp.descriptor)) for mp in
    #              points1)  # Using generator expression for performance boost.
    #     minimal_tuple = min(dists, key=lambda x: x[1])
    #     if minimal_tuple[1] < constants.SAME_DESC1:
    #         min_copy = copy.copy(minimal_tuple[0])
    #         p_copy = copy.copy(point)
    #         minimal_tuple[0].matchedPoint = point
    #         point.matchedPoint = minimal_tuple[0]
    #         # set names
    #         min_copy.name += ", " + point.name
    #         p_copy.name += ", " + minimal_tuple[0].name
    #
    #         both_points1.append(min_copy)
    #         both_points2.append(p_copy)
    #
    # return both_points1, both_points2


# TODO: separate to diff module
# TODO(olicht): make more general purpose like code
def vis_frame(frame: np.ndarray,
              points: List[Tracking_Point],
              name: Optional[Text] = "CF",
              color: Optional[Tuple[int, int, int]] = (240, 2, 22)
              ):
    """
    show the frame with all the points

    :param frame: The current frame
    :param points: All the point on screen
    :param name: Name of the image
    :param color: Color of the circles

    :return _vis_frame: The frame with all the points
    """
    _vis_frame = frame.copy()

    for pt in points:  # for each point
        circleColor = (224, 2, 24)
        if pt.isOld:
            circleColor = (24, 2, 224)
        point_copy = (int(pt.point[0]), int(pt.point[1]))
        cv.circle(_vis_frame, point_copy, constants.CIRCLE_SIZE,
                  circleColor)  # add a circle to the frame where the point is
        cv.putText(_vis_frame, pt.name, point_copy, cv.FONT_HERSHEY_DUPLEX, 0.5, color)  # add the point's name
    cv.imshow(name, _vis_frame)
    return _vis_frame


# TODO: separate to diff module
def undistort(frame: np.ndarray, num: int = 0) -> np.ndarray:
    """
    undistort a frame

    :param frame: the current frame
    :param num: number of camera

    :return: undistorted frame
    """
    # h, w = frame.shape[:2]
    # newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_params.K[num], camera_params.DistCoff[num], (w, h), 1, (w, h))

    # undistort
    frame = cv.undistort(frame, camera_params.K[num], camera_params.DistCoff[num])
    # crop the image
    # x, y, w, h = roi
    # return frame[y:y + h, x:x + w]
    return frame


# TODO: separate to diff module (also visual module)
def print_knn(frame1: np.ndarray,
              frame2: np.ndarray,
              points1: List[Tracking_Point],
              points2: List[Tracking_Point],
              matches: List[Tracking_Point],
              matchesMask: List[List[int]]
              ) -> None:
    f1 = frame1.copy()
    f2 = frame2.copy()

    # f1 = cv.resize(f1, (400, 800))
    # f2 = cv.resize(f2, (400, 800))

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    kps1 = [pt.keyPoint for pt in points1]
    kps2 = [pt.keyPoint for pt in points2]
    vis_frame = cv.drawMatchesKnn(f1, kps1, f2, kps2, matches, None, **draw_params)
    cv.imshow("Matches", vis_frame)
    return vis_frame


def point_with_loc(tpoints: List[Tracking_Point]) -> List[Tracking_Point]:
    """
    Returns only the Tracking_Point with a valid real world location

    :arg tpoints: The Tracking_Point list we want get points with real world location.

    :returns: list of Tracking_Points.
    """
    return [tp for tp in tpoints if tp.realLoc is not None]

def point_without_loc(tpoints: List[Tracking_Point]) -> List[Tracking_Point]:
    """
    Returns only the Tracking_Point without a valid real world location

    :arg tpoints: The Tracking_Point list we want get points with real world location.

    :returns: list of Tracking_Points.
    """
    return [tp for tp in tpoints if tp.realLoc is None]


def Send2PnP(tpoints: List[Tracking_Point], index: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    calls the PnP method

    :param tpoints: The Tracking_Point list we want get points with real world location.
    :param index: 1 for camera 1, 2 for camera 2

    :return:
    """
    pointsRealLoc = [tp.realLoc for tp in tpoints]
    pointsPoint = [tp.point for tp in tpoints]
    return triangulation.PnP(pointsRealLoc, pointsPoint, index)

def create_objp() -> np.ndarray:
    """
    Creates an `np.ndarray` with size `constants.BOARD_SIZE` and fills it with the 3d
    coordinates of the chessboard according to `constants.SQURESIZE`
    """
    objp = np.zeros((constants.BOARD_SIZE[0] * constants.BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:constants.BOARD_SIZE[0], 0:constants.BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= constants.SQURESIZE
    return objp


def generate_chessboard_for_camera(frame):
    """
    finds the points of the chessboard in the frame

    :param frame: the current frame
    :return points of the chessboard
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # turn frame to grayscale
    ret, corners = cv.findChessboardCorners(gray, constants.BOARD_SIZE, None)
    return ret, cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), constants.criteria)


def scanning():
    global saved_points
    my_points1 = []
    cap1 = cap2 = out1 = out2 = None
    try:
        cap1 = cv.VideoCapture(vid_file1)  # capture from camera 1 video
        cap2 = cv.VideoCapture(vid_file2)  # capture from camera 2 video

        old_frame1 = None  # old camera 1 frame
        old_frame2 = None  # old camera 2 frame

        my_points1 = []  # all of the points of camera 1
        on_screen1 = []  # all of the points on screen of camera 1

        my_points2 = []  # all of the points of camera 2
        on_screen2 = []  # all of the points on screen of camera 2

        # save the file:
        if SAVE_VIDEO:
            fourcc = cv.VideoWriter_fourcc(*'MPEG')  # 'x264' doesn't work

            WIDTH = int(cap1.get(3))  # width of video
            HEIGHT = int(cap1.get(4))  # height of video

            size1 = (WIDTH, HEIGHT)  # dimension of video

            WIDTH = int(cap2.get(3))  # width of video
            HEIGHT = int(cap2.get(4))  # height of video

            size2 = (WIDTH, HEIGHT)  # dimension of video

            out1 = cv.VideoWriter(output_name1, fourcc, 20.0, size1)  # get video
            out2 = cv.VideoWriter(output_name2, fourcc, 20.0, size2)  # get video

            size3 = (size1[0] + size2[1], HEIGHT)

            out_match = cv.VideoWriter(matching_vid, fourcc, 20.0, size3)  # get video

        for i in range(10):  # throw first 10 frames
            cap1.read()
            cap2.read()

        is_first = True
        k = 0
        tick = 0  # to count fps
        while k != ord('q') and k != ord('Q') and k != 27:  # 'Q', 'q' or ESC
            tick = time.time()
            ret1, frame1 = cap1.read()  # read next frame from camera 1
            ret2, frame2 = cap2.read()  # read next frame from camera 2

            if not ret1 or not ret2:  # if the next frame wasn't read properly
                break

            # TODO: add a section img proccess for more advance img proccess (optional)
            # frame1 = undistort(frame1, 0)  # undistort frame1
            # frame2 = undistort(frame2, 1)  # undistort frame2

            # if not is_first:
            # frame1 = cv.transpose(frame1)
            # frame2 = cv.transpose(frame2)
            #
            # frame1 = cv.flip(frame1, 0)
            # frame2 = cv.flip(frame2, 1)

            # cv.imshow("Origin", frame)  # show frame
            k = cv.waitKey(1) & 0xff

            if is_first:
                # Chessboard

                ret1, my_points1_temp = generate_chessboard_for_camera(frame1)
                ret2, my_points2_temp = generate_chessboard_for_camera(frame2)

                if not ret1 or not ret2:
                    cv.imshow("Chess1", frame1)  # show frame
                    cv.imshow("Chess2", frame2)  # show frame
                    k = cv.waitKey(1) & 0xff
                    continue

                objp = create_objp()

                # R1, T1 = triangulation.PnP(objp, my_points1_temp, 1)  # get rotation & translation from solvePnP from camera 1
                # R2, T2 = triangulation.PnP(objp, my_points2_temp, 2)  # get rotation & translation from solvePnP from camera 2

                # for p in my_points1_temp:  # add all the points from camera 1 as 'Tracking_Point to 'my_points1'
                #     kpt = points_to_keyPoints(p)
                #     d = get_sift_desc(frame1, kpt)
                #     my_points1.append(Tracking_Point("1" + next_name(), p[0], kpt, d[0], None))

                for p, rl in zip(my_points1_temp, objp):  # add all the points from camera 1 as 'Tracking_Point to 'my_points1'
                    kpt = points_to_keyPoints(p)
                    d = get_sift_desc(frame1, kpt)
                    my_points1.append(Tracking_Point("1" + next_name(), p[0], kpt, d[0], rl))

                # for p in my_points2_temp:  # add all the points from camera 2 as 'Tracking_Point to 'my_points2'
                #     kpt = points_to_keyPoints(p)
                #     d = get_sift_desc(frame2, kpt)
                #     my_points2.append(Tracking_Point("2" + next_name(), p[0], kpt, d[0], None))

                for p, rl in zip(my_points2_temp, objp):  # add all the points from camera 2 as 'Tracking_Point to 'my_points2'
                    kpt = points_to_keyPoints(p)
                    d = get_sift_desc(frame2, kpt)
                    my_points2.append(Tracking_Point("2" + next_name(), p[0], kpt, d[0], rl))

                # triangulation.find_real_points(R1, T1, R2, T2, my_points1, my_points2)

                on_screen1 = my_points1[:]  # updates 'on_screen1'
                on_screen2 = my_points2[:]  # updates 'on_screen2'
                is_first = False

            else:
                my_points1, on_screen1 = manage_points(frame1, old_frame1, my_points1, on_screen1)
                fps1 = 1.0 / (time.time() - tick)
                my_points2, on_screen2 = manage_points(frame2, old_frame2, my_points2, on_screen2)
                fps2 = 1.0 / (time.time() - tick)

                # both_cam_points(on_screen1, on_screen2)

                _frame1 = frame1.copy()
                cv.putText(_frame1, str(fps1), (0, 21), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                _frame2 = frame2.copy()
                cv.putText(_frame2, str(fps2), (0, 21), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                save_frame1 = vis_frame(_frame1, on_screen1, "Camera 1")
                save_frame2 = vis_frame(_frame2, on_screen2, "Camera 2")
                if SAVE_VIDEO:
                    out1.write(save_frame1)  # write frame
                    out2.write(save_frame2)  # write frame

                # m1, m2, m, mm = match_points(point_without_loc(on_screen1), point_without_loc(on_screen2))
                m1, m2, m, mm = match_points(on_screen1, on_screen2)
                matching_frame = print_knn(frame1, frame2, on_screen1, on_screen2, m, mm)
                # matching_frame = print_knn(frame1, frame2, point_without_loc(on_screen1), point_without_loc(on_screen2), m, mm)  # match_points and print_knn need to have the exact same on screen lists

                if SAVE_VIDEO:
                    out_match.write(matching_frame)
                R1, T1 = Send2PnP(point_with_loc(on_screen1), 1)
                R2, T2 = Send2PnP(point_with_loc(on_screen2), 2)
                triangulation.find_real_points(R1, T1, R2, T2, point_without_loc(m1), point_without_loc(m2))


            old_frame1 = frame1.copy()
            old_frame2 = frame2.copy()
    finally:
        save_points(point_with_loc(my_points1))
        print("Video is over")
        cv.destroyAllWindows()
        cap1 and cap1.release()
        cap2 and cap2.release()
        if SAVE_VIDEO:
            out1 and out1.release()
            out2 and out2.release()
            out_match and out_match.release()
    return

def chess_each_frame():
    global saved_points
    my_points1 = []
    cap1 = cap2 = out1 = out2 = None
    try:
        cap1 = cv.VideoCapture(vid_file1)  # capture from camera 1 video
        cap2 = cv.VideoCapture(vid_file2)  # capture from camera 2 video

        old_frame1 = None  # old camera 1 frame
        old_frame2 = None  # old camera 2 frame

        my_points1 = []  # all of the points of camera 1
        on_screen1 = []  # all of the points on screen of camera 1

        my_points2 = []  # all of the points of camera 2
        on_screen2 = []  # all of the points on screen of camera 2

        # save the file:
        if SAVE_VIDEO:
            fourcc = cv.VideoWriter_fourcc(*'MPEG')  # 'x264' doesn't work

            WIDTH = int(cap1.get(3))  # width of video
            HEIGHT = int(cap1.get(4))  # height of video

            size1 = (WIDTH, HEIGHT)  # dimension of video

            WIDTH = int(cap2.get(3))  # width of video
            HEIGHT = int(cap2.get(4))  # height of video

            size2 = (WIDTH, HEIGHT)  # dimension of video

            out1 = cv.VideoWriter(output_name1, fourcc, 20.0, size1)  # get video
            out2 = cv.VideoWriter(output_name2, fourcc, 20.0, size2)  # get video

            size3 = (size1[0] + size2[1], HEIGHT)

            out_match = cv.VideoWriter(matching_vid, fourcc, 20.0, size3)  # get video

        # for i in range(10):  # throw first 10 frames
        #     cap1.read()
        #     cap2.read()

        is_first = True
        k = 0
        tick = 0  # to count fps
        while k != ord('q') and k != ord('Q') and k != 27:  # 'Q', 'q' or ESC
            tick = time.time()
            ret1, frame1 = cap1.read()  # read next frame from camera 1
            ret2, frame2 = cap2.read()  # read next frame from camera 2

            if not ret1 or not ret2:  # if the next frame wasn't read properly
                break

            k = cv.waitKey(1) & 0xff

            if is_first:

                ret1, my_points1_temp = generate_chessboard_for_camera(frame1)
                ret2, my_points2_temp = generate_chessboard_for_camera(frame2)

                if not ret1 or not ret2:
                    cv.imshow("Chess1", frame1)  # show frame
                    cv.imshow("Chess2", frame2)  # show frame
                    k = cv.waitKey(1) & 0xff
                    continue

                objp = create_objp()

                # R1, T1 = triangulation.PnP(objp, my_points1_temp, 1)  # get rotation & translation from solvePnP from camera 1
                # R2, T2 = triangulation.PnP(objp, my_points2_temp, 2)  # get rotation & translation from solvePnP from camera 2

                # for p in my_points1_temp:  # add all the points from camera 1 as 'Tracking_Point to 'my_points1'
                #     kpt = points_to_keyPoints(p)
                #     d = get_sift_desc(frame1, kpt)
                #     my_points1.append(Tracking_Point("1" + next_name(), p[0], kpt, d[0], None))

                for p, rl in zip(my_points1_temp,
                                 objp):  # add all the points from camera 1 as 'Tracking_Point to 'my_points1'
                    kpt = points_to_keyPoints(p)
                    d = get_sift_desc(frame1, kpt)
                    my_points1.append(Tracking_Point("1" + next_name(), p[0], kpt, d[0], rl))

                # for p in my_points2_temp:  # add all the points from camera 2 as 'Tracking_Point to 'my_points2'
                #     kpt = points_to_keyPoints(p)
                #     d = get_sift_desc(frame2, kpt)
                #     my_points2.append(Tracking_Point("2" + next_name(), p[0], kpt, d[0], None))

                for p, rl in zip(my_points2_temp,
                                 objp):  # add all the points from camera 2 as 'Tracking_Point to 'my_points2'
                    kpt = points_to_keyPoints(p)
                    d = get_sift_desc(frame2, kpt)
                    my_points2.append(Tracking_Point("2" + next_name(), p[0], kpt, d[0], rl))

                # triangulation.find_real_points(R1, T1, R2, T2, my_points1, my_points2)

                on_screen1 = my_points1[:]  # updates 'on_screen1'
                on_screen2 = my_points2[:]  # updates 'on_screen2'
                is_first = False
            else:
                ret1, my_points1_temp = generate_chessboard_for_camera(frame1)
                ret2, my_points2_temp = generate_chessboard_for_camera(frame2)

                if not ret1 or not ret2:
                    cv.imshow("Chess1", frame1)  # show frame
                    cv.imshow("Chess2", frame2)  # show frame
                    k = cv.waitKey(1) & 0xff
                    continue

                objp = create_objp()

                R1, T1 = triangulation.PnP(objp, my_points1_temp, 1)  # get rotation & translation from solvePnP from camera 1
                R2, T2 = triangulation.PnP(objp, my_points2_temp, 2)  # get rotation & translation from solvePnP from camera 2

                my_points1, on_screen1 = manage_points(frame1, old_frame1, my_points1, on_screen1)
                fps1 = 1.0 / (time.time() - tick)
                my_points2, on_screen2 = manage_points(frame2, old_frame2, my_points2, on_screen2)
                fps2 = 1.0 / (time.time() - tick)

                _frame1 = frame1.copy()
                cv.putText(_frame1, str(fps1), (0, 21), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                _frame2 = frame2.copy()
                cv.putText(_frame2, str(fps2), (0, 21), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                save_frame1 = vis_frame(_frame1, on_screen1, "Camera 1")
                save_frame2 = vis_frame(_frame2, on_screen2, "Camera 2")
                if SAVE_VIDEO:
                    out1.write(save_frame1)  # write frame
                    out2.write(save_frame2)  # write frame

                m1, m2, m, mm = match_points(on_screen1, on_screen2)
                matching_frame = print_knn(frame1, frame2, on_screen1, on_screen2, m, mm)

                if SAVE_VIDEO:
                    out_match.write(matching_frame)
                triangulation.find_real_points(R1, T1, R2, T2, point_without_loc(m1), point_without_loc(m2))

            old_frame1 = frame1.copy()
            old_frame2 = frame2.copy()

    finally:
        save_points(point_with_loc(my_points1))
        print("Video is over")
        cv.destroyAllWindows()
        cap1 and cap1.release()
        cap2 and cap2.release()
        if SAVE_VIDEO:
            out1 and out1.release()
            out2 and out2.release()
            out_match and out_match.release()
    return


if __name__ == "__main__":
    # record_cams()
    # scanning()
    chess_each_frame()

# TODO(Tomer): we dont need to seach matching from scratch each time, we should keep a match and only search more if we need to:
#       add a minimum number of matching needed in a frame, just like points.
#       we need more points
#       save the video (capture screen program)

# TODO(Dan's advice):
#	1. do the comparicent to the descriptors with a perfect hash instead of	#	   iterating through all of them.
#	2. adding a world position the the end/beginning of the descriptor in order to
#	   handle similar points.

#  proj1 changes when it should not
#  search new points on both cameras when one updates



