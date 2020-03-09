from djitellopy import Tello
import cv2
import numpy as np
import time
import datetime
import argparse
import csv

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

SAVE_LOC1 = r'Data/drone_frame1.png'
SAVE_LOC2 = r'Data/drone_frame2.png'
SAVE_FPS = 25
frameNum=0

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

        self.save_loc_local = SAVE_LOC1

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

        #cap = cv2.VideoCapture('Data/video.mp4')
        #j = 0
        frame_read = self.tello.get_frame_read()

        should_stop = False
        imgCount = 0
        OVERRIDE = True
        oSpeed = args.override_speed
        tDistance = args.distance
        self.tello.get_battery()

        sec = 0
        index = 0
        ideal_positions = self.getRealIdealPos()

        # Safety Zone X
        szX = args.saftey_x

        # Safety Zone Y
        szY = args.saftey_y

        if args.debug:
            print("DEBUG MODE ENABLED!")

        self.init()

        print("Finished Init")

        try:
            while not should_stop:
                self.update()
                #self.tello.get_battery()

                if frame_read.stopped:
                    frame_read.stop()
                    break

                theTime = str(datetime.datetime.now()).replace(':', '-').replace('.', '_')

                if time.time() - (1 / SAVE_FPS) > sec:
                    # firas_frameRet, firas_frame = cap.read()
                    frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
                    frameRet = frame_read.frame
                    # frame = np.rot90(frame)
                    imgCount += 1
                    cv2.imwrite(self.save_loc_local, frame)
                    #cv2.imwrite('Data/Video/frame' + str(j) + '.jpg', frame)
                    #j += 1
                    sec = time.time()
                    self.save_loc_local = SAVE_LOC2 if self.save_loc_local == SAVE_LOC1 else SAVE_LOC1

                time.sleep(1 / FPS)
                # firas demo day
                # if self.get_track_state()== 0:
                #     print("Going up.. searching")
                #     self.for_down_velocity = int(2)
                #     # self.for_down_velocity = int(2)
                #     self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                #                                self.yaw_velocity)

                # Listen for key presses
                k = cv2.waitKey(20)

                if k == ord('b'):
                    self.tello.get_battery()


                # b to save checkpoint
                if k == ord('n'):
                    print("********************************Checkpoint Saved******************************************")
                    checkpoints.append(self.getRealTimePos())
                    np.savetxt(r"Data/checkpoints.csv", checkpoints, delimiter=" ")

                    # print("********************************CHECKPOINT SAVED******************************************")
                    # with open(r"Data/normalized.csv", "r") as f:
                    #     for x in range(5):
                    #         normalized.append(self.getRealTimePos())
                    #     normalized.append(zeros)
                    #     np.savetxt(r"Data/normalized.csv", normalized, delimiter=" ")

                    #checkpoints.append(self.normalize())
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
                        #print("******************OVERRIDE ENABLED*******************")
                    else:
                        OVERRIDE = False
                        self.send_rc_control = True
                        #print("******************OVERRIDE DISABLED******************")

                if OVERRIDE:
                    # S & W to fly forward & back
                    if k == ord('w'):
                        #print("******************OVERRIDE FORWARD*******************")
                        self.send_rc_control = True
                        self.tello.send_rc_control(0, 15, 0, 0)
                        self.send_rc_control = False
                    elif k == ord('s'):
                        #print("******************OVERRIDE BACK*******************")
                        self.send_rc_control = True
                        self.tello.send_rc_control(0, -15, 0, 0)
                        self.send_rc_control = False
                    else:
                        self.for_back_velocity = 0

                    # a & d to pan left & right
                    if k == ord('d'):
                        #print("******************OVERRIDE RIGHT*******************")
                        self.send_rc_control = True
                        self.tello.send_rc_control(15, 0, 0, 0)
                        self.send_rc_control = False
                    elif k == ord('a'):
                        #print("******************OVERRIDE LEFT*******************")
                        self.send_rc_control = True
                        self.tello.send_rc_control(-15, 0, 0, 0)
                        self.send_rc_control = False
                    else:
                        self.yaw_velocity = 0

                    # Q & E to fly up & down
                    if k == ord('e'):
                        #print("******************OVERRIDE UP*******************")
                        self.send_rc_control = True
                        self.tello.send_rc_control(0, 0, 15, 0)
                        self.send_rc_control = False
                    elif k == ord('q'):
                        #print("******************OVERRIDE DOWN*******************")
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
        frameNum=0
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
                self.save_loc_local = SAVE_LOC2 if self.save_loc_local == SAVE_LOC1 else SAVE_LOC1

            time.sleep(1 / FPS)

            k = cv2.waitKey(20)

            if k == ord('t'):
                if not args.debug:
                    print("Taking Off")
                    self.tello.takeoff()
                    self.tello.get_battery()
                self.send_rc_control = True

            if k == ord('s'):
                print("********************************Frame Saved******************************************")
                cv2.imwrite("frames/frame" + str(frameNum) + ".png", frameRet)
                frameNum = frameNum + 1

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

    # def rotate_right(self, mul=1):
    #     self.yaw_velocity = mul * int(20)
    #     frame_read = self.tello.get_frame_read()
    #     time_spinning = time.time()
    #     sec = 0
    #     while time.time() - time_spinning < 11:
    #         self.update()
    #
    #         if frame_read.stopped:
    #             frame_read.stop()
    #             break
    #
    #         if time.time() - (1 / SAVE_FPS) > sec:
    #             frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
    #             frameRet = frame_read.frame
    #             cv2.imwrite(self.save_loc_local, frame)
    #             sec = time.time()
    #             self.save_loc_local = SAVE_LOC2 if self.save_loc_local == SAVE_LOC1 else SAVE_LOC1
    #
    #         time.sleep(1 / FPS)
    #
    #         k = cv2.waitKey(20)
    #
    #         if k == ord('t'):
    #             if not args.debug:
    #                 print("Taking Off")
    #                 self.tello.takeoff()
    #                 self.tello.get_battery()
    #             self.send_rc_control = True
    #
    #         # Press L to land
    #         if k == ord('l'):
    #             if not args.debug:
    #                 print("Landing")
    #                 self.tello.land()
    #             self.send_rc_control = False
    #
    #         # Display the resulting frame
    #         cv2.imshow(f'Tello Tracking...', frameRet)
    #     self.yaw_velocity = 0

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

    def getRealIdealPos(self):
        positions = []
        with open(r"Data/idealpos.csv", "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                positions.append(np.array(row[:3], np.float32))
        return positions


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()