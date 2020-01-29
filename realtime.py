#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import csv
import time
from typing import List, Tuple, Text, AnyStr, Union, Optional, Iterator

import cv2 as cv
import numpy as np
import pickle
import triangulation

import camera_params
import constants

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import av
import tellopy

# sift = cv.xfeatures2d.SIFT_create(constants.FEATURES_TO_TRACK)

# orb = cv.ORB_create(
#     nfeatures=constants.FEATURES_TO_TRACK,
#     scaleFactor=1.2,
#     nlevels=8,
#     WTA_K=4
# )

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

SAVE_VIDEO = False

video_name = "new"
vid_file1 = "outputs/" + video_name + "1.avi"
output_name = "outputs/" + video_name + "_points" + ".avi"

vid_file2 = "outputs/" + video_name + "2.avi"

saved_points = []
curr_num = 0
min_featured = constants.MIN_FOLLOW_POINTS


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
    :param matchedPointName: the name of the matched point in an old frame
    """

    def __init__(self, name=None, point=(1, 1), keyPoint=None, descriptor=None, realLoc=None):
        self.name = name
        self.point = tuple(point[0] if len(point) == 1 else point)
        self.keyPoint = keyPoint
        self.descriptor = descriptor
        self.realLoc = realLoc
        self.matchedPointName = ""
        self.isOld = False


# TODO
# def find_features_ORB(frame):
#   kp = orb.detect(frame, None)
#   kp = kp[:, :, 3] # Ignore the appearance of the same point 3 times
#   points = [k.pt for k in kp]
#   return points, kp

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


def points_to_keyPoints(points, size=10):  # TODO(88roy88): Docstring
    """
    returns keyPoints of points

    :param points: list of points as tuples/lists
    :param size: points diameter

    :return: list of keyPoints
    """
    return [cv.KeyPoint(point[0], point[1], size) for point in points]


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

    new_points = new_points.reshape([-1, 1, 2])  # -1 is any number, 2 is for x,y # TODO: Reomove this 1
    new_points = convert_to_tuple_list(new_points.tolist(), idx=0)  # Normalize to array of tuples
    return new_points, st


index_params = dict(algorithm=constants.FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params, search_params)  # TODO: Understand


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
        with open(f"{basename}.csv", "a+") as f:
            # Add unvisited points to file
            for tp in Tracking_Points:
                f.write("{name},{pt},{kp},{desc},{x},{y},{z}\n".format(
                    name=tp.name,
                    pt=tp.point,
                    kp=tp.keyPoint,
                    desc=tp.descriptor,
                    x=tp.realLoc[0],
                    y=tp.realLoc[1],
                    z=tp.realLoc[2]
                )
                )
                saved_points.append(tp)
    elif mode == "pickle":  # saves in pickle format
        with open(f"{basename}.pickle", "w") as f:
            pickle.dump(Tracking_Point, f)
            saved_points += Tracking_Points
    # TODO: Add XML option
    else:
        print("Unknown saving format \"{0}\". Not saving the points...".format(mode))


def load_points(pointFile="Data//pointData", descFile="Data//descriptorsData.xml"):
    """
    Load external points

    :arg mode: (Optional, default: `constants.DEFAULT_SAVE_MODE`)
    :arg basename: (Optional, default: "model") The basename for the file.

    :returns: List with the points.
    """
    points = []
    descriptors = []
    extern_points = []

    with open(f"{pointFile}.csv", "r") as f:
        spamreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in spamreader:
            # read from row
            points.append((row[0], row[1], row[2]))

    i = 0
    cv_file = cv.FileStorage(descFile, cv.FILE_STORAGE_READ)
    matrix = cv_file.getNode("desc0").mat()
    while matrix is not None:
        descriptors.append(matrix)
        i += 1
        matrix = cv_file.getNode("desc" + str(i)).mat()
    cv_file.release()

    for point, desc in zip(points, descriptors):
        extern_points.append(Tracking_Point(realLoc=point, descriptor=desc))

    return extern_points


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
            if point is not mp and hamming_dist(point.descriptor,
                                                mp.descriptor) < constants.HAMMING:  # if 'mp' isn't 'point' and its distance to their descriptors is less than SAME_DESC
                print(point.name + " removed DESC by " + ("Data" if mp.name is None else mp.name))
                point.realLoc = mp.realLoc
                merge_on_screen.add(point)  # add the points that are on screen
                point.isOld = True
                return False
        return True

    my_points_temp = filter(filterFunc,
                            my_points_temp)  # puts each point in the list only if 'filterFunc' returns true, aka isn't already on screen
    return my_points_temp, merge_on_screen


def hamming_dist(desc1, desc2):
    x = np.int64(desc1) ^ np.int64(desc2)
    x -= (x >> 1) & 0x5555555555555555
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
    return np.sum((np.uint64(x * 0x0101010101010101) & np.uint64(0xffffffffffffffff)) >> 56)


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
    # my_points_temp, merge_on_screen1 = merge_same_point_by_pos(my_points_temp, on_screen)  # merge by distance
    my_points_temp, merge_on_screen2 = merge_same_point_by_desc(my_points_temp, my_points)  # merge by descriptors
    merge_on_screen2 = []
    my_points_temp = list(my_points_temp)
    on_screen_temp = list(merge_on_screen1.union(merge_on_screen2))
    on_screen_temp = [item for item in on_screen_temp if item not in on_screen]
    for pt in my_points_temp:
        pt.name = next_name()

    # my_points += my_points_temp  # updates 'my_points'
    on_screen = my_points_temp + on_screen_temp + [item for item in on_screen if
                                                   item not in on_screen_temp]  # updates 'on_screen' - 'my_points_temp': new on screen, 'on_screen_temp': old on screen

    return my_points, on_screen


# to move to better place in code
def record_cam():
    """
    Record video from the two cameras and save it
    """
    cap = out1 = None  # For code-style check
    try:
        cap = cv.VideoCapture(0)  # capture camera

        WIDTH = int(cap.get(3))  # width of video
        HEIGHT = int(cap.get(4))  # height of video

        size = (WIDTH, HEIGHT)  # dimension of video

        fourcc = cv.VideoWriter_fourcc(*'MPEG')  # 'x264' doesn't work

        out1 = cv.VideoWriter(vid_file1, fourcc, 20.0, size)  # get video

        while cap.isOpened():
            _, frame = cap.read()  # read frame from cam

            out1.write(frame)  # write frame

            cv.imshow('frame', frame)  # show frame

            k = cv.waitKey(1) & 0xFF
            if k == ord('q') or k == ord('Q') or k == 27:  # 'Q', 'q', or Esc in order to stop showing the frames
                break
    finally:
        cap and cap.release()
        out1 and out1.release()
        cv.destroyAllWindows()

# to move to better place in code
def record_two_cams():
    """
    Record video from the two cameras and save it
    """
    cap1 = out1 = None  # For code-style check
    cap2 = out2 = None  # For code-style check
    try:
        cap1 = cv.VideoCapture(1)  # capture camera
        cap2 = cv.VideoCapture(2)  # capture camera

        WIDTH = int(cap1.get(3))  # width of video
        HEIGHT = int(cap1.get(4))  # height of video

        size1 = (WIDTH, HEIGHT)  # dimension of video

        WIDTH = int(cap2.get(3))  # width of video
        HEIGHT = int(cap2.get(4))  # height of video

        size2 = (WIDTH, HEIGHT)  # dimension of video

        fourcc = cv.VideoWriter_fourcc(*'MPEG')  # 'x264' doesn't work

        out1 = cv.VideoWriter(vid_file1, fourcc, 30.0, size1)  # get video

        out2 = cv.VideoWriter(vid_file2, fourcc, 30.0, size2)  # get video

        while cap1.isOpened():
            _, frame1 = cap1.read()  # read frame from cam

            _, frame2 = cap2.read()  # read frame from cam

            out1.write(frame1)  # write frame

            out2.write(frame2)  # write frame

            cv.imshow('frame1', frame1)  # show frame

            cv.imshow('frame2', frame2)  # show frame

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
                  on_screen: List[Tracking_Point]
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
    st = None
    try:
        if old_frame is not None:  # if it isn't the first frame
            points = [mp.point for mp in on_screen]
            new_points, st = opticFlow(old_frame, frame, points)  # get the updated points and the status
            # if sum(st) >= min_featured:  # if there are enough points on screen
            new_on_screen = []

            for mp, np, s in zip(on_screen, new_points, st):  # updates the points
                mp.point = np
                if s:  # if the point is on the screen, add it to 'new_on_screen'
                    new_on_screen.append(mp)
            on_screen = new_on_screen
    except cv.error as e:  # fixed?
        if str(e).find("1244") > 0:
            st = [0]
            print("Avoided not enough points")
        else:
            raise

    if old_frame is None or sum(st) < min_featured:  # if it is the first frame or there aren't enough points on screen
        ps, kps, desc = find_features_ORB(
            frame)  # ps - featured points, kps - the matching keyPoints, d - the matching descriptors
        # because we don't have enough points on screen (or non if its the first frame) we need to find new featured points (can be different from the ones we have now)

        desc = [] if desc is None else desc

        if old_frame is None:  # first frame
            my_points_temp = [Tracking_Point(next_name(), p, k, d, [-1, -1, -1]) for p, k, d in zip(ps, kps,
                                                                                                    desc)]  # create a 'Tracking_Point' variable to ach featured point, puts all in a list
        else:
            my_points_temp = [Tracking_Point("Temp", p, k, d, [-1, -1, -1]) for p, k, d in zip(ps, kps,
                                                                                               desc)]  # create a 'Tracking_Point' variable to ach featured point, puts all in a list

        _, on_screen = merge_same_point(my_points_temp, my_points, on_screen)
        on_screen = match_points(on_screen, my_points)

        min_featured = min(max(len(on_screen), constants.MAX_FOLLOW_POINTS),
                           constants.MIN_FOLLOW_POINTS)  # if after adding all the points there isn't enough, we will lower the threshold in order to stabilize

    return my_points, on_screen


# bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=False)

# index_params = dict(algorithm=constants.FLANN_INDEX_KDTREE, trees=5)
index_params = dict(algorithm=constants.FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,  # 20
                    multi_probe_level=1)  # 2
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params, search_params)  # TODO: Understand


def match_points(points: List[Tracking_Point],
                 point_set: List[Tracking_Point],
                 DESC_LIMIT: Optional[int] = 80):
    """
    Takes two sets of points and find a "quick" matching between them

    :arg points: The point set we would like to compare
    :arg point_set: The set to whom we compare to
    :arg DESC_LIMIT: (Optional, default: 100, too high to catch anything) The max distance between paired points
    """

    if len(points) == 0:
        return []

    desc1 = np.array([pt.descriptor for pt in points])  # list of descriptors of the input points
    desc2 = np.array([pt.descriptor[0] for pt in point_set])  # list ofq descriptors of the saved set of points

    matches = flann.knnMatch(desc1, desc2, 2)

    good_matches = []

    for m, n in matches:
        if m.distance*constants.LIMIT < n.distance:
            if m.distance <= DESC_LIMIT:
                good_matches.append(m)

    good_matches.sort(key=lambda x: x.distance)

    for m in range(min(constants.BEST_FEATURES, len(good_matches))):
        points[good_matches[m].queryIdx].realLoc = point_set[good_matches[m].trainIdx].realLoc
        points[good_matches[m].queryIdx].isOld = True

    return points


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

    # Filter points that are too close and old
    points_filter = []
    for i in range(len(points)):
        dist = min(
            [100] + [np.linalg.norm(np.array(point.point) - np.array(points[i].point)) for point in points_filter])
        if points[i].isOld and dist > 10:
            points_filter.append(points[i])

    print("Before", len(points))
    print("After", len(points_filter))

    for pt in points:  # for each point
        if pt.isOld:
            circleColor = (24, 2, 224)
            point_copy = (int(pt.point[0]), int(pt.point[1]))
            real_loc = str(pt.realLoc)
            cv.circle(_vis_frame, point_copy, constants.CIRCLE_SIZE,
                      circleColor)  # add a circle to the frame where the point is
            cv.putText(_vis_frame, pt.name + " " + real_loc, point_copy, cv.FONT_HERSHEY_DUPLEX, 0.5,
                       color)  # add the point's name
    cv.imshow(name, _vis_frame)

    # old_points = np.array([pt.realLoc for pt in points if pt.isOld]).transpose()
    # print(old_points)
    # plot_location(old_points[0], old_points[1], old_points[2])
    return _vis_frame


# TODO: separate to diff module
def undistort(frame: np.ndarray, num: int = 0) -> np.ndarray:
    """
    undistort a frame

    :param frame: the current frame
    :param num: number of camera

    :return: undistorted frame
    """
    frame = cv.undistort(frame, camera_params.K[num], camera_params.DistCoff[num])
    return frame


def point_with_loc(tpoints: List[Tracking_Point]) -> List[Tracking_Point]:
    """
    Returns only the Tracking_Point with a valid real world location

    :arg tpoints: The Tracking_Point list we want get points with real world location.

    :returns: list of Tracking_Points.
    """
    return [tp for tp in tpoints if (tp.realLoc != [-1, -1, -1] and tp.realLoc is not None)]


def Send2PnP(tpoints: List[Tracking_Point]):
    """
    calls the PnP method

    :param tpoints: The Tracking_Point list we want get points with real world location.

    :return:
    """
    pointsRealLoc = [tp.realLoc for tp in tpoints]
    pointsPoint = [tp.point for tp in tpoints]
    if len(tpoints) < 4:
        print("Not Enough Points - Less than 3")
        return None
    return triangulation.PnP(pointsRealLoc, pointsPoint, 1)


# def get_camera_position(on_screen):
#     R, T = Send2PnP(point_with_loc(on_screen), 1)


def scanning():
    cap = None
    try:
        cap = cv.VideoCapture(vid_file1)  # capture from camera 1 video

        old_frame = None  # old camera 1 frame

        my_points = load_points()  # all of the points of the camera
        on_screen = []  # all of the points on screen of the camera

        # save the file:
        if SAVE_VIDEO:
            fourcc = cv.VideoWriter_fourcc(*'MPEG')  # 'x264' doesn't work

            WIDTH = int(cap.get(3))  # width of video
            HEIGHT = int(cap.get(4))  # height of video

            size = (WIDTH, HEIGHT)  # dimension of video

            out = cv.VideoWriter(output_name, fourcc, 20.0, size)  # get video

        ret = True
        first = True

        num_of_frames = 0
        t_x = []
        t_y = []
        t_z = []
        X = []
        Y = []
        Z = []

        is_first = True  # first frame
        k = 0
        tick = 0
        while k != ord('q') and k != ord('Q') and k != 27:  # 'Q', 'q' or ESC
            tick = time.time()
            ret, frame = cap.read()  # read next frame from camera

            if not ret:  # if the next frame wasn't read properly
                break

            # TODO: add a section img proccess for more advance img proccess (optional)
            frame = undistort(frame, 0)  # undistort frame

            # cv.imshow("Origin", frame)  # show frame
            k = cv.waitKey(1) & 0xff

            _, on_screen = manage_points(frame, old_frame, my_points, on_screen)
            fps = 1.0 / (time.time() - tick)
            _frame = frame.copy()
            cv.putText(_frame, f"{fps:#07.3f}", (0, 21), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
            save_frame = vis_frame(_frame, on_screen, "Points")
            if SAVE_VIDEO:
                out.write(save_frame)  # write frame

            time.sleep(10)

            cam_pos = triangulation.get_camera_position(Send2PnP(point_with_loc(on_screen)))
            if cam_pos is not None:
                t_x.append(cam_pos[0][0])
                t_y.append(cam_pos[1][0])
                t_z.append(cam_pos[2][0])
                num_of_frames += 1
                if num_of_frames >= 30:
                    X.append(np.median(t_x))
                    Y.append(np.median(t_y))
                    Z.append(np.median(t_z))
                    t_x = []
                    t_y = []
                    t_z = []
                    num_of_frames = 0

            is_first = False
            old_frame = frame.copy()
    finally:
        plot_location(X, Y, Z)
        print("Video is over")
        cv.destroyAllWindows()
        cap and cap.release()
        if SAVE_VIDEO:
            out and out.release()
    return


def find_scale_factor(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    if not ret:
        print("No Chessboard found in first frame, scale factor set to default")
        scale_factor = 1
    else:
        scale_factor = 1
    return scale_factor


def plot_location(X, Y, Z):
    X = X.astype(np.float)
    Y = Y.astype(np.float)
    Z = Z.astype(np.float)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    x = np.array(X)
    y = np.array(Y)
    z = np.array(Z)

    ax.plot(x, y, z)

    plt.show()


def drone():
    try:
        drone = tellopy.Tello()
        drone.log.set_level(2)
        drone.connect()
        drone.start_video()

        # container for processing the packets into frames
        container = av.open(drone.get_video_stream())
        video_st = container.streams.video[0]

        old_frame = None  # old camera 1 frame

        my_points = load_points()  # all of the points of the camera
        on_screen = []  # all of the points on screen of the camera

        # save the file:
        if SAVE_VIDEO:
            fourcc = cv.VideoWriter_fourcc(*'MPEG')  # 'x264' doesn't work

            WIDTH = int(960)  # width of video
            HEIGHT = int(720)  # height of video

            size = (WIDTH, HEIGHT)  # dimension of video

            out = cv.VideoWriter(output_name, fourcc, 20.0, size)  # get video

        ret = True

        num_of_frames = 0
        t_x = []
        t_y = []
        t_z = []
        X = []
        Y = []
        Z = []

        is_first = True  # first frame
        k = 0
        tick = 0
        for packet in container.demux((video_st,)):
            for frame in packet.decode():
                tick = time.time()
                frame = cv.cvtColor(np.array(frame.to_image()), cv.COLOR_RGB2BGR)

                # cv.imshow("Origin", frame)  # show frame
                k = cv.waitKey(1) & 0xff

                if k == ord('q') or k == ord('Q') or k == 27:
                    return

                _, on_screen = manage_points(frame, old_frame, my_points, on_screen)
                fps = 1.0 / (time.time() - tick)
                _frame = frame.copy()
                cv.putText(_frame, f"{fps:#07.3f}", (0, 21), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                save_frame = vis_frame(_frame, on_screen, "Points")
                if SAVE_VIDEO:
                    out.write(save_frame)  # write frame

                cam_pos = triangulation.get_camera_position(Send2PnP(point_with_loc(on_screen)))
                if cam_pos is not None:
                    t_x.append(cam_pos[0][0])
                    t_y.append(cam_pos[1][0])
                    t_z.append(cam_pos[2][0])
                    num_of_frames += 1
                    if num_of_frames >= 30:
                        X.append(np.median(t_x))
                        Y.append(np.median(t_y))
                        Z.append(np.median(t_z))
                        t_x = []
                        t_y = []
                        t_z = []
                        num_of_frames = 0

                is_first = False
                old_frame = frame.copy()
    finally:
        print(len(X))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X, Y, Z, c='r', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        x = np.array(X)
        y = np.array(Y)
        z = np.array(Z)

        ax.plot(x, y, z)

        plt.show()
        print("Video is over")
        drone.quit()
        cv.destroyAllWindows()
        if SAVE_VIDEO:
            out and out.release()


def main():
    record_cam()
    #record_two_cams()
    # scanning()
     #drone()


if __name__ == "__main__":
    main()
