from djitellopy import Tello
import cv2
import numpy as np
import time
import argparse
import csv
from typing import List, Tuple, Text, AnyStr, Union, Optional, Iterator
import pickle
import triangulation
import constants
import matplotlib.pyplot as plt
import camera_params
import av
import tellopy

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

# Speed of the drone

S = 15
SS = 15
UDOffset = 150
SAVE_VIDEO = False

# These are the values in which kicks in speed up mode, as of now,
# this hasn't been finalized or fine tuned so be careful

# Tested are 3, 4, 5
acc = [500, 250, 250, 150, 110, 70, 50]

# Frames per second
FPS = 25
dimensions = (960, 720)

checkpoints = []
normalized = []
zeros = [0] * 3

CLOSE_THRESHOLD = 0.05

SAVE_FPS = 25
cam_pos = None


class FrontEnd(object):

    def __init__(self):
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 20
        self.theta = 0

        self.send_rc_control = False

    def run(self):
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
        #ourframe = frame_read.frame
        cap = frame_read.cap

        sec = 0
        should_stop = False
        OVERRIDE = True
        oSpeed = args.override_speed
        tDistance = args.distance
        self.tello.get_battery()
        index = 0
        ideal_positions = self.getRealIdealPos()

        # Safety Zone X
        szX = args.saftey_x

        # Safety Zone Y
        szY = args.saftey_y

        #self.init()

        try:
            while not should_stop:
                self.update()

                if frame_read.stopped:
                    frame_read.stop()
                    break

                old_frame = None  # old camera 1 frame
                my_points = load_points()  # all of the points of the camera
                my_des = np.array([pt.descriptor[0] for pt in my_points])  # list of descriptors of the saved set of points
                on_screen = []  # all of the points on screen of the camera
                points_3d = []
                points_2d = []
                ret = True
                first = True
                k = 0
                first_frame = True
                while k != ord('q') and k != ord('Q') and k != 27:  # 'Q', 'q' or ESC
                    tick = time.time()

                    # Get frame:
                    ourframe = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
                    if ourframe is None:
                        break
                    #ourframe = undistort(ourframe, 0)  # undistort frame
                    k = cv2.waitKey(1) & 0xff
                    orb = cv2.ORB_create()

                    # find the keypoints with ORB
                    kp = orb.detect(ourframe, None)

                    # compute the descriptors with ORB
                    kp, des = orb.compute(ourframe, kp)

                    # create BFMatcher object
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                    # Match descriptors.
                    matches = bf.match(des, my_des)

                    # Sort them in the order of their distance.
                    matches = sorted(matches, key=lambda x: x.distance)

                    for m in range(10):
                        points_3d.append(my_points[matches[m].trainIdx].realLoc)
                        points_2d.append(kp[matches[m].queryIdx].pt)

                  #  ret, rv, tv = cv2.solvePnPRansac(points_3d, points_2d, camera_params.TELLO, camera_params.DistCoff)

                    ret, rv, tv = cv2.solvePnP(np.array(points_3d, dtype=np.float32),
                              np.array(points_2d, dtype=np.float32),
                              np.array(camera_params.TELLO),
                              camera_params.DistCoff)
                    thistuple = (rv, tv)
                    cam_pos = triangulation.get_camera_position(thistuple)

                    _, on_screen = manage_points(ourframe, old_frame, my_points, on_screen)

                    fps = 1.0 / (time.time() - tick)
                    _frame = ourframe.copy()
                    cv2.putText(_frame, f"{fps:#07.3f}", (0, 21), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                    save_frame = vis_frame(_frame, on_screen, "Points")

                    time.sleep(0.1)
                    # rv,tv = Send2PnP(point_with_loc(on_screen))
                    # print(rv)
                    # print(tv)
                    #cam_pos = triangulation.get_camera_position(Send2PnP(point_with_loc(on_screen)))
                    cam_pos = triangulation.get_camera_position(Send2PnP(point_with_loc(on_screen)))
                    if cam_pos is not None and first_frame is True:
                         print(cam_pos[0][0]," ",cam_pos[1][0]," ",cam_pos[2][0])
                    first = False

                    old_frame = ourframe.copy()

                    frame = cv2.cvtColor(ourframe, cv2.COLOR_BGR2RGB)
                    frameRet = ourframe

                    if k == ord('b'):
                        self.tello.get_battery()

                    # b to save checkpoint
                    if k == ord('n'):
                        print(
                            "********************************Checkpoint Saved******************************************")
                        checkpoints.append(self.getRealTimePos())
                        np.savetxt(r"Data/checkpoints.csv", checkpoints, delimiter=" ")

                        # print("********************************CHECKPOINT SAVED******************************************")
                        # with open(r"Data/normalized.csv", "r") as f:
                        #     for x in range(5):
                        #         normalized.append(self.getRealTimePos())
                        #     normalized.append(zeros)
                        #     np.savetxt(r"Data/normalized.csv", normalized, delimiter=" ")

                        # checkpoints.append(self.normalize())
                        # np.savetxt(r"Data/checkpoints.csv", normalized, delimiter=" ")

                    # Press 0 to set distance to 0
                    if k == ord('0'):
                        if not OVERRIDE:
                            print("Distance = 0")
                            tDistance = 0

                    # Press 1 to set distance to 1
                    if k == ord('1'):
                        if OVERRIDE:
                            oSpeed = 1
                        else:
                            print("Distance = 1")
                            tDistance = 1

                    # Press 2 to set distance to 2
                    if k == ord('2'):
                        if OVERRIDE:
                            oSpeed = 2
                        else:
                            print("Distance = 2")
                            tDistance = 2

                    # Press 3 to set distance to 3
                    if k == ord('3'):
                        if OVERRIDE:
                            oSpeed = 3
                        else:
                            print("Distance = 3")
                            tDistance = 3

                    # Press 4 to set distance to 4
                    if k == ord('4'):
                        if not OVERRIDE:
                            print("Distance = 4")
                            tDistance = 4

                    # Press 5 to set distance to 5
                    if k == ord('5'):
                        if not OVERRIDE:
                            print("Distance = 5")
                            tDistance = 5

                    # Press 6 to set distance to 6
                    if k == ord('6'):
                        if not OVERRIDE:
                            print("Distance = 6")
                            tDistance = 6

                    # Press T to take off
                    if k == ord('t'):
                        if not args.debug:
                            print("Taking Off")
                            self.tello.takeoff()
                            self.tello.get_battery()
                        self.send_rc_control = True

                    # Press L to land
                    if k == ord('l'):
                        if not args.debug:
                            print("Landing")
                            self.tello.land()
                        self.send_rc_control = False

                    # Press X for controls override
                    if k == ord('x'):
                        if not OVERRIDE:
                            OVERRIDE = True
                            # print("******************OVERRIDE ENABLED*******************")
                        else:
                            OVERRIDE = False
                            self.send_rc_control = True
                            # print("******************OVERRIDE DISABLED******************")

                    if OVERRIDE:
                        # S & W to fly forward & back
                        if k == ord('w'):
                            # print("******************OVERRIDE FORWARD*******************")
                            self.send_rc_control = True
                            self.tello.send_rc_control(0, 15, 0, 0)
                            self.send_rc_control = False
                        elif k == ord('s'):
                            # print("******************OVERRIDE BACK*******************")
                            self.send_rc_control = True
                            self.tello.send_rc_control(0, -15, 0, 0)
                            self.send_rc_control = False
                        else:
                            self.for_back_velocity = 0

                        # a & d to pan left & right
                        if k == ord('d'):
                            # print("******************OVERRIDE RIGHT*******************")
                            self.send_rc_control = True
                            self.tello.send_rc_control(15, 0, 0, 0)
                            self.send_rc_control = False
                        elif k == ord('a'):
                            # print("******************OVERRIDE LEFT*******************")
                            self.send_rc_control = True
                            self.tello.send_rc_control(-15, 0, 0, 0)
                            self.send_rc_control = False
                        else:
                            self.yaw_velocity = 0

                        # Q & E to fly up & down
                        if k == ord('e'):
                            # print("******************OVERRIDE UP*******************")
                            self.send_rc_control = True
                            self.tello.send_rc_control(0, 0, 15, 0)
                            self.send_rc_control = False
                        elif k == ord('q'):
                            # print("******************OVERRIDE DOWN*******************")
                            self.send_rc_control = True
                            self.tello.send_rc_control(0, 0, -15, 0)
                            self.send_rc_control = False
                        else:
                            self.up_down_velocity = 0

                        # c & z to fly left & right
                        if k == ord('c'):
                            self.left_right_velocity = int(S * oSpeed)
                        elif k == ord('z'):
                            self.left_right_velocity = -int(S * oSpeed)
                        else:
                            self.left_right_velocity = 0

                    # Quit the software
                    if k == 27:
                        should_stop = True
                        break

                    tello_pos = self.getRealTimePos()

                    if tello_pos is not None:
                        print(tello_pos)

                        trans_vec = tello_pos - ideal_positions[index]

                        if np.all(np.abs(trans_vec) < CLOSE_THRESHOLD):
                            index += 1
                            print("----------------------------------------------------------------------------")
                            print("NEXT CHECKPOINT")
                            print("----------------------------------------------------------------------------")
                            try:
                                trans_vec = tello_pos - ideal_positions[index]
                            except Exception as e:
                                print("Reached the end destination")
                                return

                        trans_vec = self.vec_after_rotation(trans_vec)

                        if trans_vec[0] > CLOSE_THRESHOLD:
                            self.left_right_velocity = - int(S * oSpeed)
                            # self.left_right_velocity = - int(5)
                        elif trans_vec[0] < -CLOSE_THRESHOLD:
                            self.left_right_velocity = int(S * oSpeed)
                            # self.left_right_velocity = int(5)
                        else:
                            self.left_right_velocity = 0

                        if trans_vec[1] > CLOSE_THRESHOLD:
                            self.for_back_velocity = - int(SS * oSpeed)
                        elif trans_vec[1] < -CLOSE_THRESHOLD:
                            self.for_back_velocity = int(SS * oSpeed)
                        else:
                            self.for_back_velocity = 0

                        if trans_vec[2] > CLOSE_THRESHOLD:
                            self.up_down_velocity = - int(SS * oSpeed)
                        elif trans_vec[2] < -CLOSE_THRESHOLD:
                            self.up_down_velocity = int(SS * oSpeed)
                        else:
                            self.up_down_velocity = 0

                        if OVERRIDE:
                            show = "OVERRIDE: {}".format(oSpeed)
                            dCol = (255, 255, 255)
                        else:
                            show = "AI: {}".format(str(tDistance))

                        # Display the resulting frame
                        cv2.imshow(f'Tello Tracking...', frameRet)

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

    def battery(self):
        return self.tello.get_battery()[:2]

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

    def init(self):
        frame_read = self.tello.get_frame_read()
        first_pos = self.getRealTimePos()
        sec = 0
        # print("Taking Off")
        # self.tello.takeoff()
        # self.tello.get_battery()
        # while not self.check_orb_control(first_pos):
        while not self.check_orb_control(first_pos):
            self.update()

            if frame_read.stopped:
                frame_read.stop()
                break

            if time.time() - (1 / SAVE_FPS) > sec:
                frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
                frameRet = frame_read.frame
                cv2.imwrite(self.save_loc_local, frame)
                sec = time.time()
                # self.save_loc_local = SAVE_LOC2 if self.save_loc_local == SAVE_LOC1 else SAVE_LOC1

            time.sleep(1 / FPS)

            k = cv2.waitKey(20)

            if k == ord('t'):
                if not args.debug:
                    print("Taking Off")
                    self.tello.takeoff()
                    self.tello.get_battery()
                self.send_rc_control = True

            # Press L to land
            if k == ord('l'):
                if not args.debug:
                    print("Landing")
                    self.tello.land()
                self.send_rc_control = False

            # Display the resulting frame
            cv2.imshow(f'Tello Tracking...', frameRet)

            #self.for_down_velocity = int(10)
            self.for_back_velocity = -int(8)
            self.left_right_velocity = int(0)
        #self.for_down_velocity = int(0)
        self.for_back_velocity = int(0)

    def rotate_left(self):
        self.rotate_right(mul=-1)

    def vec_after_rotation(self, vec):
        x = vec[0] * np.cos(self.theta) - vec[1] * np.sin(self.theta)
        y = vec[0] * np.sin(self.theta) + vec[1] * np.cos(self.theta)
        return np.array([x, y, vec[2]], np.float32)

    # firas demo day
    # def get_track_state(self):
    #     f = open("/home/fares/src/orbslam2/ORB_SLAM2/trackstate.txt", "r")
    #     for x in f:
    #             state = int(x)
    #     return state

    def getRealTimePos(self):
        if cam_pos is not None:
            cam_pos_trans = np.transpose(cam_pos)
            return cam_pos_trans
        return None

    def check_orb_control(self, prev_pos):
        if np.all(np.equal(self.getRealTimePos(), prev_pos)):
            return False
        return True

    def getRealIdealPos(self):
        positions = []
        with open(r"Data/idealpos.csv", "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                positions.append(np.array(row[:3], np.float32))
        return positions

# sift = cv.xfeatures2d.SIFT_create(constants.FEATURES_TO_TRACK)

# orb = cv.ORB_create(
#     nfeatures=constants.FEATURES_TO_TRACK,
#     scaleFactor=1.2,
#     nlevels=8,
#     WTA_K=4
# )

orb = cv2.ORB_create(
    nfeatures=constants.FEATURES_TO_TRACK,
    scaleFactor=1.2,
    nlevels=8,
#    edgeThreshold=31,
#    firstLevel=0,
#    WTA_K=2,
#    scoreType=cv2.ORB_HARRIS_SCORE,
#    patchSize=31,
    fastThreshold=20
)

SAVE_VIDEO = False

video_name = "outputVid"
output_name = "finalvideo.avi"
vid_file1 = "/home/fares/src/orbslam2/ORB_SLAM2/mapsandvideos/2/demoVid.avi"

vid_file2 = "outputs/" + video_name + "2.avi"
# END ASALA WAAD

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
    :return: ORB descriptor
    """

    kp, desc = orb.detectAndCompute(frame, None)
    points = [k.pt for k in kp]
    return points, kp, desc



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
    new_points, st, err = cv2.calcOpticalFlowPyrLK(
        old_frame,
        frame,
        points,
        None,
        **constants.OPTIC_FLOW_lk_params
    )

    new_points = new_points.reshape([-1, 1, 2])  # -1 is any number, 2 is for x,y # TODO: Reomove this 1
    new_points = convert_to_tuple_list(new_points.tolist(), idx=0)  # Normalize to array of tuples
    return new_points, st


#index_params = dict(algorithm=constants.FLANN_INDEX_KDTREE, trees=5)
#search_params = dict(checks=50)  # or pass empty dictionary
index_params = dict(algorithm=constants.FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,  # 20
                    multi_probe_level=1)  # 2
search_params = dict(checks=50)  # or pass empty dictionary                 // try to test with checks=100 // Higher vales gives better precsion , but also takes more time

flann = cv2.FlannBasedMatcher(index_params, search_params)  # TODO: Understand


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
    pointFile = "Data/pointData"
    descFile = "Data/descriptorsData.xml"
    points = []
    descriptors = []
    extern_points = []

    with open(f"{pointFile}.csv", "r") as f:
        spamreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in spamreader:
            # read from row
            points.append((row[0], row[1], row[2]))

    i = 0
    cv_file = cv2.FileStorage(descFile, cv2.FILE_STORAGE_READ) # opens descriptor file
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
                scPt.point = point.point
                merge_on_screen.add(scPt)
                return False
        return True

    my_points_temp = filter(filterFunc,
                            my_points_temp)  # puts each point in the list only if 'filterFunc' returs true, aka isn't very close to another point
    return my_points_temp, merge_on_screen


# TODO(olicht): fix to use match point now
def merge_same_Rpoint_by_desc(my_points_temp: List[Tracking_Point],
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
        return True  # isnt already on screen

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
   # my_points_temp, merge_on_screen1 = merge_same_point_by_pos(my_points_temp, on_screen)  # merge by distance
  #  print (my_points_temp)
    my_points_temp, merge_on_screen1 = merge_same_point_by_desc(my_points_temp, my_points)  # merge by descriptors
        # in my_points_temp now we have only the points that werent deleted (new points )
        # in merge_on_screen1 we have all the points only on screen
        # in my_points we have the same points but we updated the isOld feature
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
    except cv2.error as e:  # fixed?
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

# index_params = dict(algorithm=constants.FLANN_INDEX_KDTREE, trees=5) // need to check if this works better
# index_params = dict(algorithm=constants.FLANN_INDEX_LSH,
#                     table_number=6,  # 12
#                     key_size=12,  # 20
#                     multi_probe_level=1)  # 2
# search_params = dict(checks=50)  # or pass empty dictionary                 // try to test with checks=100 // Higher vales gives better precsion , but also takes more time

#flann = cv2.FlannBasedMatcher(index_params, search_params)  # TODO: Understand


def match_points(points: List[Tracking_Point],
                 point_set: List[Tracking_Point],
                 DESC_LIMIT: Optional[int] = 80):
    """
    Takes two sets of points and find a "quick" matching between them
    :arg points: The point set we would like to compare     // on_screen
    :arg point_set: The set to whom we compare to          //my_points
    :arg DESC_LIMIT: (Optional, default: 100, too high to catch anything) The max distance between paired points
    """


                # need to check BFmatcher
    if len(points) == 0:
        return []

    desc1 = np.array([pt.descriptor for pt in points])  # list of descriptors of the input points
    desc2 = np.array([pt.descriptor[0] for pt in point_set])  # list of descriptors of the saved set of points
    #desc1.convertTo (desc1,CV_32F)

    matches = flann.knnMatch(desc1, desc2, 2)

    good_matches = []

    for m, n in matches:
        if m.distance < n.distance*0.7:
             if m.distance <= DESC_LIMIT:       # need to check if realy this line needed
                good_matches.append(m)
    #      if m.distance*constants.LIMIT < n.distance:    if m.distance < 0.7*n.distance
    good_matches.sort(key=lambda x: x.distance)
    #print(good_matches)
    for m in range(min(constants.BEST_FEATURES, len(good_matches))):
        points[good_matches[m].queryIdx].realLoc = point_set[good_matches[m].trainIdx].realLoc     # need to understand
        points[good_matches[m].queryIdx].isOld = True                                              # need to understand

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
        if points[i].isOld:
            points_filter.append(points[i])
#and dist > 10 :
    # print("Before", len(points))
    # print("After", len(points_filter))

    for pt in points:  # for each point
        if pt.isOld:
            circleColor = (24, 2, 224)
            point_copy = (int(pt.point[0]), int(pt.point[1]))
            real_loc = str(pt.realLoc)
            cv2.circle(_vis_frame, point_copy, constants.CIRCLE_SIZE,
                      circleColor)  # add a circle to the frame where the point is
            cv2.putText(_vis_frame, pt.name + " " + real_loc, point_copy, cv2.FONT_HERSHEY_DUPLEX, 0.5,
                       color)  # add the point's name
    cv2.imshow(name, _vis_frame)

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
    #frame = cv2.undistort(frame, camera_params.NEW_TELLO, camera_params.DistCoff)
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
        cap = cv2.VideoCapture(vid_file1)  # capture from camera 1 video

        old_frame = None  # old camera 1 frame

        my_points = load_points()  # all of the points of the camera
        on_screen = []  # all of the points on screen of the camera

        # save the file:
        if SAVE_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # 'x264' doesn't work

            WIDTH = int(cap.get(3))  # width of video
            HEIGHT = int(cap.get(4))  # height of video

            size = (WIDTH, HEIGHT)  # dimension of video

            out = cv2.VideoWriter(output_name, fourcc, 20.0, size)  # get video

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
            k = cv2.waitKey(1) & 0xff

            _, on_screen = manage_points(frame, old_frame, my_points, on_screen)

            fps = 1.0 / (time.time() - tick)
            _frame = frame.copy()
            cv2.putText(_frame, f"{fps:#07.3f}", (0, 21), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
            save_frame = vis_frame(_frame, on_screen, "Points")
            if SAVE_VIDEO:
                out.write(save_frame)  # write frame

            time.sleep(0.1)

            cam_pos = triangulation.get_camera_position(Send2PnP(point_with_loc(on_screen)))

            if cam_pos is not None:
                t_x.append(cam_pos[0][0])
                t_y.append(cam_pos[1][0])
                t_z.append(cam_pos[2][0])
                print (cam_pos[0][0])
                print (cam_pos[1][0])
                print (cam_pos[2][0])
                #cam_plot_line(cam_pos[0][0],cam_pos[1][0],cam_pos[2][0])
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

       # plot_location(X, Y, Z)
        print("Video is over")
        cv2.destroyAllWindows()
        cap and cap.release()
        if SAVE_VIDEO:
            out and out.release()
    return



def plot_location(X, Y, Z):
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    Z = np.array(Z, dtype=np.float32)
    # X = X.astype(np.float)
    # Y = Y.astype(np.float)
    # Z = Z.astype(np.float)

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

def main():
    frontend = FrontEnd()
    frontend.run()


if __name__ == '__main__':
    main()
