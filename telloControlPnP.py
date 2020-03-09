from djitellopy import Tello
import cv2
import numpy as np
import time
import datetime
import argparse
import csv
from typing import List, Tuple, Text, AnyStr, Union, Optional, Iterator
import CPPCORESET
import camera_params
import constants
import generatelines
import os
import matplotlib.pyplot as plt

# standard arg parse stuff
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='** = required')
parser.add_argument('-d', '--distance', type=int, default=3,
                    help='use -d to change the distance of the drone. Range 0-6')
parser.add_argument('-sx', '--saftey_x', type=int, default=100,
                    help='use -sx to change the saftey bound on the x axis . Range 0-480')
parser.add_argument('-sy', '--saftey_y', type=int, default=55,
                    help='use -sy to change the saftey bound on the y axis . Range 0-360')
parser.add_argument('-os', '--override_speed', type=int, default=1,
                    help='use -os to change override speed. Range 0-3')
parser.add_argument('-ss', "--save_session", action='store_true',
                    help='add the -ss flag to save your session as an image sequence in the Sessions folder')
parser.add_argument('-D', "--debug", action='store_true',
                    help='add the -D flag to enable debug mode. Everything works the same, but no commands will be sent to the drone')

args = parser.parse_args()

orb = cv2.ORB_create(
    nfeatures=constants.FEATURES_TO_TRACK,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20
)

min_featured = 20
num_best_points = 100
frameRate = 30
CAMERA_CALIBRATION = camera_params.TELLO
DIST_COFF = camera_params.DistCoff
# Choose True to use coresets
USE_CORESET = False
# Choose True to use opticflow
USE_OPTICFLOW = True

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(240, 180),
                 maxLevel=7,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Speed of the drone
S = 7
UDOffset = 150

# Frames per second
FPS = 25
dimensions = (960, 720)
pointstodraw = []
ideal_positions_ar = []
checkpoints = []
normalized = []
zeros = [0] * 3

CLOSE_THRESHOLD = 0.015

SAVE_LOC1 = r'Data/drone_frame1.png'
SAVE_LOC2 = r'Data/drone_frame2.png'
SAVE_FPS = 12

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('window.avi', fourcc, 25, (960, 720))


class FrontEnd(object):

    def __init__(self):
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        self.theta = 0

        self.send_rc_control = False

        self.save_loc_local = SAVE_LOC1

    def run(self):
        OPTICFLOW_FLAG = False
        frameK = False
        # TODO: change to base frame ( frame from building the map (last frame ?) )
        old_frame = None
        cap = None

        # TODO: check if declared right
        indexes = np.array([])
        weights = np.array([])
        points_3d = np.array([])
        points_2d = np.array([])
        points_3d_cor = np.array([])
        points_2d_cor = np.array([])
        points_2d_optic = np.array([])
        points_3d_optic = np.array([])
        frameNum = 0

        # TODO: should add CSV reader and writer ?
        # TODO: check if readed write
        # 3D model points
        ModelPoints = load_points()

        # Descriptors of the saved set of points
        ModelDescriptors = np.array([pt.descriptor[0] for pt in ModelPoints])

        global checkpoints
        global normalized
        global zeros

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        frame_read = self.tello.get_frame_read()
        # cap = cv2.VideoCapture(2)  # capture camera
        should_stop = False
        imgCount = 0
        OVERRIDE = True
        oSpeed = args.override_speed
        tDistance = args.distance
        self.tello.get_battery()

        sec = 0
        index = 0
        ideal_positions = self.getRealIdealPos()
        plt.axes()
        for i in range(len(ideal_positions)):
            ideal_positions_ar.append(self.vec_after_rotation(ideal_positions[i]))
        for i in range(len(ideal_positions_ar)):
            index = (i + 1) % len(ideal_positions_ar)
            line = plt.Line2D((ideal_positions_ar[i][0], ideal_positions_ar[index][0]),
                              (ideal_positions_ar[i][1], ideal_positions_ar[index][1]), lw=1.5)
            plt.gca().add_line(line)
        plt.axis('scaled')

        if args.debug:
            print("DEBUG MODE ENABLED!")

        try:
            while not should_stop:
                self.update()

                if frame_read.stopped:
                    frame_read.stop()
                    break

                if time.time() - (1 / SAVE_FPS) > sec:
                    ourframe = frame_read.frame
                    video.write(ourframe)
                    # _,ourframe = cap.read()
                    imgCount += 1
                    # cv2.imwrite(self.save_loc_local, ourframe)

                    sec = time.time()
                    self.save_loc_local = SAVE_LOC2 if self.save_loc_local == SAVE_LOC1 else SAVE_LOC1

                time.sleep(1 / FPS)
                frameK = False
                if (frameNum is frameRate) or (OPTICFLOW_FLAG and 'inliers_points_2d' in vars() and len(inliers_points_2d) < min_featured):
                    frameK = True
                    frameNum = 0

                # if first frame or K frame => OPTICFLOW_FLAG = false
                if (OPTICFLOW_FLAG and USE_OPTICFLOW) is False or frameK is True:
                    points_3d, points_2d = get_points(
                        ourframe,
                        old_frame,
                        ModelPoints,
                        ModelDescriptors
                    )

                    # TODO: run ransac before running optimal then run optimal on inliers only
                    # Generate lines
                    lines = generatelines.generatelines(points_2d, CAMERA_CALIBRATION)

                    if not USE_CORESET:

                        try:
                            _, _, _, inliers = cv2.solvePnPRansac(np.array(points_3d), np.array(points_2d),
                                                                  np.array(CAMERA_CALIBRATION), np.array(DIST_COFF))
                            inliers_points_3d = []
                            inliers_lines = []
                            inliers_points_2d = []
                            for i in range(len(inliers)):
                                inliers_lines.append(np.array(lines[inliers[i][0]]))
                                inliers_points_3d.append(np.array(points_3d[inliers[i][0]]))
                                inliers_points_2d.append(np.array(points_2d[inliers[i][0]]))
                        except:
                            frameNum = frameRate
                            print("ERROR: cant detect new points")
                            continue
                        inliers_points_3d = np.array(inliers_points_3d)
                        inliers_lines = np.array(inliers_lines)
                        R, t = optimalPnP(inliers_points_3d, inliers_lines)
                        R[0, 1] = -R[0, 1]



                    if USE_CORESET:
                        points_3d_cor, points_2d_cor, lines_cor, weights, indexes = createCoreset(
                            ourframe,
                            old_frame,
                            points_3d,
                            points_2d,
                            lines)

                        R, t = weighted_optimalPnP(points_3d_cor, lines_cor, weights, indexes)
                    OPTICFLOW_FLAG = True
                    cam_pos = np.dot(np.transpose(R), t)
                    # if cam_pos is not None:
                    #    print(cam_pos[0], " ", cam_pos[1], " ", cam_pos[2])

                elif (OPTICFLOW_FLAG and USE_CORESET) is True:
                    # use points_3d_cor, points_2d_cor with opticflow
                    points_2d_optic, st = opticFlow(old_frame, ourframe, points_2d_cor)
                    lines_cor_optic = generatelines.generatelines(points_2d_optic, CAMERA_CALIBRATION)
                    R, t = weighted_optimalPnP(points_3d_cor, lines_cor_optic, weights, indexes)
                    R[0, 1] = -R[0, 1]
                elif OPTICFLOW_FLAG is True:
                    # use points_3d, points_2d with opticflow
                    points_2d_optic, st = opticFlow(old_frame, ourframe, inliers_points_2d)
                    lines_optic = generatelines.generatelines(points_2d_optic, CAMERA_CALIBRATION)

                    inliers_3d = []
                    for i in range(len(inliers_points_3d)):
                        if st[i] == 1:
                            inliers_3d.append(inliers_points_3d[i])
                    inliers_points_3d = inliers_3d

                    R, t = optimalPnP(inliers_points_3d, lines_optic)
                    R[0, 1] = -R[0, 1]
                    inliers_points_2d = points_2d_optic
                cam_pos = np.dot(np.transpose(R), t)

                # if cam_pos is not None:
                #     print(cam_pos[0], " ", cam_pos[1], " ", cam_pos[2])
                # try:
                #     with open(r"Data/pos1.csv", "r") as f:
                #         np.savetxt(r"Data/pos1.csv", cam_pos, delimiter=" ", newline=" ")
                # except:
                #     with open(r"Data/pos2.csv", "r") as f:
                #         np.savetxt(r"Data/pos2.csv", cam_pos, delimiter=" ", newline=" ")

                old_frame = ourframe.copy()
                frameNum += 1

                # Listen for key presses
                k = cv2.waitKey(30) & 0xff
                if k == ord('b'):
                    self.tello.get_battery()

                if k == ord('s'):
                    print("********************************Frame Saved******************************************")
                    cv2.imwrite("frames/frame" + str(frameNum) + ".png", ourframe)

                # b to save checkpoint
                if k == ord('n'):
                    print("********************************Checkpoint Saved******************************************")
                    checkpoints.append(cam_pos)
                    np.savetxt(r"Data/checkpoints.csv", checkpoints, delimiter=" ")

                # Press T to take off
                if k == ord('t'):
                    if not args.debug:
                        print("Taking Off")
                        self.tello.takeoff()
                        # time.sleep(1)
                        # self.tello.rotate_counter_clockwise(45)
                        time.sleep(4)
                        self.tello.get_battery()
                        frameNum = frameRate
                    self.send_rc_control = True

                # Press L to land
                if k == ord('l'):
                    if not args.debug:
                        print("Landing")
                        self.tello.land()
                    self.send_rc_control = False

                # Quit the software
                if k == 27:
                    should_stop = True
                    break

                trans_vec = cam_pos - ideal_positions[index]
                trans_vec[2] = 0
                # print(np.abs(trans_vec))
                print(cam_pos)
                if np.all(np.abs(trans_vec) < CLOSE_THRESHOLD):
                    index += 1
                    pointstodraw.append(self.vec_after_rotation(cam_pos))
                    print("----------------------------------------------------------------------------")
                    print("NEXT CHECKPOINT")
                    print("----------------------------------------------------------------------------")
                    try:
                        trans_vec = cam_pos - ideal_positions[index]
                    except Exception as e:
                        print("Reached the end destination")
                        plt.axes()
                        for i in range(len(pointstodraw)):
                            index = (i + 1) % len(pointstodraw)
                            line = plt.Line2D((pointstodraw[i][0], pointstodraw[index][0]),
                                              (pointstodraw[i][1], pointstodraw[index][1]), lw=1.5)
                            plt.gca().add_line(line)
                        plt.axis('scaled')
                        return

                trans_vec = self.vec_after_rotation(trans_vec)

                if trans_vec[0] > CLOSE_THRESHOLD:
                    # self.left_right_velocity = int(8)
                    self.left_right_velocity = int(S * oSpeed)
                elif trans_vec[0] < -CLOSE_THRESHOLD:
                    # self.left_right_velocity = - int(8)
                    self.left_right_velocity = - int(S * oSpeed)
                else:
                    self.left_right_velocity = 0

                if trans_vec[1] > CLOSE_THRESHOLD:
                    self.for_back_velocity = - int(S * oSpeed)
                elif trans_vec[1] < -CLOSE_THRESHOLD:
                    self.for_back_velocity = int(S * oSpeed)
                else:
                    self.for_back_velocity = 0

                if trans_vec[2] > CLOSE_THRESHOLD:
                    self.up_down_velocity = - int(S * oSpeed)
                elif trans_vec[2] < -CLOSE_THRESHOLD:
                    self.up_down_velocity = int(S * oSpeed)
                else:
                    self.up_down_velocity = 0

                if OVERRIDE:
                    show = "OVERRIDE: {}".format(oSpeed)
                    dCol = (255, 255, 255)
                else:
                    show = "AI: {}".format(str(tDistance))


            # On exit, print the battery
            self.tello.get_battery()

            # When everything done, release the capture
            cv2.destroyAllWindows()

            # Call it always before finishing. I deallocate resources.
            self.tello.end()
        finally:
            time.sleep(1 / FPS)
            print("Landing")
            self.tello.land()

    def getRealIdealPos(self):
        positions = []
        with open(r"Data/idealpos.csv", "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                positions.append(np.array(row[:3], np.float32))
        return positions

    def battery(self):
        return self.tello.get_battery()[:2]

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

    def rotate_left(self):
        self.rotate_right(mul=-1)

    def vec_after_rotation(self, vec):
        x = vec[0] * np.cos(self.theta) - vec[1] * np.sin(self.theta)
        y = vec[0] * np.sin(self.theta) + vec[1] * np.cos(self.theta)
        return np.array([x, y, vec[2]], np.float32)

    def getRealTimePos(self):
        pos = []
        try:
            with open(r"Data/realtimepos1.csv", "r") as f:
                reader = csv.reader(f, delimiter=" ")
                row = next(reader)
                pos.append(-np.float64(row[0]))
                pos.append(-np.float64(row[2]))
                pos.append(row[1])
                # self.theta = next(reader)[0]

        except:
            with open(r"Data/realtimepos2.csv", "r") as f:
                reader = csv.reader(f, delimiter=" ")
                row = next(reader)
                pos.append(-np.float64(row[0]))
                pos.append(-np.float64(row[2]))
                pos.append(row[1])
                # self.theta = next(reader)[0]

        return np.array(pos, np.float32)

    def check_orb_control(self, prev_pos):
        if np.all(np.equal(self.getRealTimePos(), prev_pos)):
            return False
        return True

#
# def get_points(current_frame,
#                old_frame,
#                ModelPoints,
#                ModelDescriptors):
#     points3d = []
#     points2d = []
#     print("***********************Detecting new points***********************")
#     orb = cv2.ORB_create(nfeatures=2000)
#     # find the keypoints and compute the orb descriptors
#     kp, des = orb.detectAndCompute(current_frame, None)
#     FLANN_INDEX_LSH = 6
#     index_params = dict(algorithm=FLANN_INDEX_LSH, trees=12)
#     search_params = dict(checks=30)
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     # matches = flann.knnMatch(ModelDescriptors, des,k=2)
#     matches = flann.knnMatch(des,ModelDescriptors,k=2)
#     for m, n in matches:
#         if m.distance < 0.7* n.distance:
#             points3d.append(ModelPoints[m.trainIdx].realLoc)
#             # points3d.append(ModelPoints[m.queryIdx].realLoc)
#             points2d.append(kp[m.queryIdx].pt)
#             # points2d.append(kp[m.trainIdx].pt)
#     return points3d, points2d


def get_points(current_frame,
               old_frame,
               ModelPoints,
               ModelDescriptors,
               ):
    # TODO: check if declared right
    points3d = []
    points2d = []
    kp = []
    des = []
    bf = []
    matches = []
    print("***********************Detecting new points***********************")

    orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2,
                         patchSize=31, fastThreshold=20)

    # find the keypoints with ORB
    kp = orb.detect(current_frame, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(current_frame, kp)
    # TODO:change to this ?
    # kp, des = orb.detectAndCompute(current_frame, None)
    # TODO: check if size of desp > size of our full data

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # TODO: should be matched with previous frame too ?
    # bf.match(query image (model), train image(frame))
    matches = bf.match(ModelDescriptors, des)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    frame_num = 0
    # TODO: check if range < matches
    # TODO: change 2- to const
    for m in range(num_best_points):
        points3d.append(ModelPoints[matches[m].queryIdx].realLoc)
        points2d.append(kp[matches[m].trainIdx].pt)
    # TODO: add to check correctnes
    # img3 = cv2.drawMatches(old_frame, kp1, current_frame, kp2, matches[:num_best_points], None, flags=2)
    # plt.imshow(img3)
    # plt.show()
    return points3d, points2d


# Returns matched coreset 2d points and 3d points
def createCoreset(current_frame, old_frame, points3D, points2D, Lines):
    points3d_coreset = []
    points2d_coreset = []
    lines_coreset = []

    indexArray, weightArray = CPPCORESET.run(len(points3D), points3D, points2D, Lines)

    # create coreset points:
    for i in range(len(indexArray)):
        points3d_coreset.append(points3D[indexArray[i]])
        points2d_coreset.append(points2D[indexArray[i]])
        lines_coreset.append(Lines[indexArray[i]])

    return points3d_coreset, points2d_coreset, lines_coreset, weightArray, indexArray


# Optimal PnP
def optimalPnP(points3D, lines):
    with open('OMat.txt', 'w') as f:
        f.write("%s 9 30\n" % len(points3D))
        for i in range(len(points3D)):
            for item in lines[i]:
                f.write("%s " % item)
            f.write(" 0 0 0 ")
            for item in points3D[i]:
                f.write("%s " % item)
            f.write("\n")
        f.write("0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \n")
        for item in points3D:
            f.write("1 ")

    os.system("./pnp_o OMat.txt > /dev/null")

    with open('OPnP_R.txt', 'r') as f:
        rv1 = [[float(num) for num in line.split(',')] for line in f]
    with open('OPnP_t.txt', 'r') as f:
        t = f.read().split(',')
    tv1 = []
    for el in t:
        tv1.append(float(el))

    tv = np.array(tv1);
    rv = np.array(rv1);
    return rv, tv


def weighted_optimalPnP(points3D, lines, weight, index):
    with open('coresetOMat.txt', 'w') as f:
        f.write("%s 9 30\n" % len(points3D))
        for i in range(len(points3D)):
            for item in lines[i]:
                f.write("%s " % item)
            f.write(" 0 0 0 ")
            for item in points3D[i]:
                f.write("%s " % item)
            f.write("\n")
        f.write("0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \n")
        # for item in points3D:
        #     # f.write("1 ")
        for i in range(len(points3D)):
            f.write(str(weight[i]) + ' ')
    os.system("./pnp_o coresetOMat.txt")

    with open('OPnP_R.txt', 'r') as f:
        rv1 = [[float(num) for num in line.split(',')] for line in f]
    with open('OPnP_t.txt', 'r') as f:
        t = f.read().split(',')
    tv1 = []
    for el in t:
        tv1.append(float(el))

    tv = np.array(tv1);
    rv = np.array(rv1);
    return rv, tv


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
            points.append((float(row[0]), float(row[1]), float(row[2])))

    i = 0
    cv_file = cv2.FileStorage(descFile, cv2.FILE_STORAGE_READ)
    matrix = cv_file.getNode("desc0").mat()
    while matrix is not None:
        descriptors.append(matrix)
        i += 1
        matrix = cv_file.getNode("desc" + str(i)).mat()
    cv_file.release()

    for point, desc in zip(points, descriptors):
        extern_points.append(Tracking_Point(realLoc=point, descriptor=desc))

    return extern_points


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
    mask = np.zeros_like(frame)
    points = np.array(points).astype('float32')

    new_points: np.ndarray
    st = err = None
    new_points, st, err = cv2.calcOpticalFlowPyrLK(
        old_frame,
        frame,
        points,
        None,
        **lk_params
    )


    good_new = []
    good_old = []
    # Select good points
    for i in range(len(new_points)):
        if st[i] == 1:
            good_new.append(new_points[i])
            good_old.append(points[i])

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), (0,255,0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0,0,255), -1)
    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)

    good_new = np.array(good_new)
    good_new = good_new.reshape([-1, 1, 2])  # -1 is any number, 2 is for x,y # TODO: Reomove this 1
    good_new = convert_to_tuple_list(good_new.tolist(), idx=0)  # Normalize to array of tuples

    return good_new, st


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
