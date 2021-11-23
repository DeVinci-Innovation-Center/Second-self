import threading
import time
from queue import Queue

import numpy as np
from cv2 import cv2

from components.draw_pose import DrawPose
from settings import (
    FPS,
    DEBUG_DATA,
    DEBUG_TIME,
    WINDOW,
    WIDTH,
    HEIGHT,
    RESOLUTION_X,
    RESOLUTION_Y,
    OFFSET_X,
    OFFSET_Y,
    DIMENSION_X,
    DIMENSION_Y,
)
from utils import BODY_LINKS, normalize_data

data_queue = Queue(1)
global_data = {
    "face_mesh": [],
    "body_pose": [],
    "left_hand_pose": [],
    "left_hand_sign": [],  # [SIGN, probability]
    "right_hand_pose": [],
    "right_hand_sign": [],  # [SIGN, probability]
    "eyes": [],
}


class FrameProvider(threading.Thread):
    """
    (Thread)
    * Reads frames using the 2 previous classes' functions
    * and stores them into global variables. (global) depth will be none if
    * the camera isn't the Intel D435
    """

    def __init__(self, threadID, cam):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.cam = cam
        self.color = None
        self.depth = None

    def run(self):
        """Update frames"""
        print(
            """
        -------------------------------------
        Frame provider running
        -------------------------------------
        """
        )
        while 1:
            self.color, self.depth = self.cam.next_frame()  # Updates global variales
            time.sleep(1 / (2 * FPS))  # Runs faster to be sure to get the current frame


class BodyProvider(threading.Thread):
    """
    (Thread)
    * Gets body pose from lightweight pose estimation
    """

    def __init__(self, threadID, feed):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.feed = feed

    def run(self):
        import components.get_body_pose as gbp

        print(
            """
        -------------------------------------
        Body pose running
        -------------------------------------
        """
        )
        pose = gbp.init()

        while 1:
            start_t = time.time()  # Used to mesure the elapsed time of each loop

            if self.feed.color is not None and self.feed.depth is not None:
                data = gbp.find_body_pose(
                    pose, self.feed.color, WINDOW
                )  # Infer on image, return keypoints

                global_data["body_pose"] = data

                if not data_queue.empty():
                    data_queue.get()
                data_queue.put(global_data)

            end_t = time.time()
            dt = max(1 / FPS - (end_t - start_t), 0.001)
            time.sleep(
                dt
            )  # Sleeps for 1/FPS (e.g: 33ms if FPS=60) if the code is fast enough, else 1 ms


class HandsProvider(threading.Thread):
    """
    (Thread)
    * Hands from mediapipe
    ! Only one instance from mediapipe can run
    """

    def __init__(self, threadID, feed):
        import components.get_hand_gesture as gh
        import components.get_hand_sign as ghs

        threading.Thread.__init__(self)
        self.threadID = threadID
        self.feed = feed
        self.hands = gh.init()
        self.sign_provider = ghs.init()

    def run(self):
        import components.get_hand_gesture as gh

        # import get_hand_sign as ghs

        print(
            """
        -------------------------------------
        Hands provider running
        -------------------------------------
        """
        )

        while 1:
            start_t = time.time()

            if self.feed.color is not None:
                data = gh.find_hand_pose(self.hands, self.feed.color, WINDOW)

                if bool(data):
                    global_data["right_hand_pose"] = data[
                        0
                    ]  # Arbitrary, for testing purposes
                    # global_data["right_hand_sign"] = ghs.find_gesture(
                    #     self.sign_provider,
                    #     normalize_data(self.data[0])
                    # )

                    if not data_queue.empty():
                        data_queue.get()
                    data_queue.put(global_data)
            end_t = time.time()
            dt = max(1 / FPS - (end_t - start_t), 0.001)
            time.sleep(dt)


class FaceProvider(threading.Thread):
    """
    (Thread)
    * Face from mediapipe
    ! Only one instance from mediapipe can run
    """

    def __init__(self, threadID, feed):
        import components.get_face_mesh as gf

        threading.Thread.__init__(self)
        self.threadID = threadID
        self.feed = feed
        self.faces = gf.init()

    def run(self):
        import components.get_face_mesh as gf

        print(
            """
        -------------------------------------
        Face mesh running
        -------------------------------------
        """
        )

        while 1:
            start_t = time.time()

            if self.feed.color is not None:
                data = gf.find_face_mesh(self.faces, self.feed.color, WINDOW)

                global_data["face_mesh"] = data

                if not data_queue.empty():
                    data_queue.get()
                data_queue.put(global_data)

            end_t = time.time()
            dt = max(1 / FPS - (end_t - start_t), 0.001)
            time.sleep(dt)


class HolisticProvider(threading.Thread):
    """
    (Thread)
    * Body pose from mediapipe
    ! Only one instance from mediapipe can run
    """

    def __init__(self, threadID, feed):
        import components.get_hand_sign as ghs
        import components.get_holistic as gh

        threading.Thread.__init__(self)
        self.threadID = threadID
        self.feed = feed
        self.holistic = gh.init()
        self.sign_provider = ghs.init()
        self.paused = False

    def run(self):
        # * Home made hand signs : https://github.com/Thomas-Jld/gesture-recognition
        import components.get_hand_sign as ghs
        import components.get_holistic as gh
        from components.get_reflection import project

        print(
            """
        -------------------------------------
        Holistic running
        -------------------------------------
        """
        )

        while 1:
            if not self.paused:
                start_t = time.time()

                if self.feed.color is not None and self.feed.depth is not None:
                    data = gh.find_all_poses(self.holistic, self.feed.color, WINDOW)
                    if DEBUG_DATA:
                        print(data)

                    if bool(data["body_pose"]):
                        flag_1 = time.time()

                        eyes = data["body_pose"][0][0:2]

                        body = project(
                            points=data["body_pose"],
                            eyes_position=eyes,
                            video_provider=self.feed.cam,
                            depth_frame=self.feed.depth,
                            depth_radius=2,
                        )
                        global_data["body_pose"] = body

                        global_data["right_hand_pose"] = project(
                            points=data["right_hand_pose"],
                            eyes_position=eyes,
                            video_provider=self.feed.cam,
                            depth_frame=self.feed.depth,
                            depth_radius=2,
                            ref=body[15],
                        )

                        if len(data["right_hand_pose"]) > 0:
                            global_data["right_hand_sign"] = ghs.find_gesture(
                                self.sign_provider,
                                normalize_data(data["right_hand_pose"], WIDTH, HEIGHT),
                            )

                        global_data["left_hand_pose"] = project(
                            points=data["left_hand_pose"],
                            eyes_position=eyes,
                            video_provider=self.feed.cam,
                            depth_frame=self.feed.depth,
                            depth_radius=2,
                            ref=body[16],
                        )

                        if len(data["left_hand_pose"]) > 0:
                            global_data["left_hand_sign"] = ghs.find_gesture(
                                self.sign_provider,
                                normalize_data(data["left_hand_pose"], WIDTH, HEIGHT),
                            )

                        global_data["face_mesh"] = project(
                            points=data["face_mesh"],
                            eyes_position=eyes,
                            video_provider=self.feed.cam,
                            depth_frame=self.feed.depth,
                            depth_radius=2,
                            ref=body[2],
                        )

                        flag_2 = time.time()

                        if not data_queue.empty():
                            data_queue.get()
                        data_queue.put(global_data)

                        if DEBUG_TIME:
                            print(f"Inference: {(flag_1 - start_t)*1000} ms")
                            print(f"Projection: {(flag_2 - flag_1)*1000} ms")
                            print(f"Adding to queue: {(time.time() - flag_2)*1000} ms")

                end_t = time.time()

                if DEBUG_TIME:
                    print(f"Total inference time: {(end_t - start_t)*1000}ms")
                    print(f"FPS: {int(1/(end_t - start_t))}")

                dt = max(1 / FPS - (end_t - start_t), 0.0001)
                time.sleep(dt)
            else:
                time.sleep(5)


class PifpafProvider(threading.Thread):
    """
    (Thread)
    * Body pose from pifpaf
    TODO: Select the person
    """

    def __init__(self, threadID, feed):
        # * Home made hand signs : https://github.com/Thomas-Jld/gesture-recognition
        import components.get_hand_sign as ghs
        import components.get_pifpaf as gpp

        threading.Thread.__init__(self)
        self.threadID = threadID
        self.feed = feed
        self.processor = gpp.init()
        self.sign_provider = ghs.init()
        self.paused = False

    def run(self):
        import components.get_hand_sign as ghs
        import components.get_pifpaf as gpp
        from components.get_reflection import project

        print(
            """
        -------------------------------------
        Pifpaf running
        -------------------------------------
        """
        )

        while 1:
            if not self.paused:
                start_t = time.time()
                if self.feed.color is not None and self.feed.depth is not None:
                    data = gpp.find_all_poses(self.processor, self.feed.color, WINDOW)

                    if bool(data["body_pose"]):
                        eyes = data["body_pose"][0]

                        body = project(
                            data["body_pose"],
                            eyes,
                            self.feed.cam,
                            self.feed.depth,
                            4,
                        )
                        global_data["body_pose"] = body

                        global_data["right_hand_pose"] = project(
                            data["right_hand_pose"],  # Data to project
                            eyes,  # POV for reflection
                            self.feed.cam,  # Data from the camera
                            self.feed.depth,  # Depth map
                            2,  # Size of the are to sample from
                            body[9][
                                -1
                            ],  # (Optionnal) Distance to use instead of the real one
                        )

                        if len(data["right_hand_pose"]) > 0:
                            global_data["right_hand_sign"] = ghs.find_gesture(
                                self.sign_provider,
                                normalize_data(data["right_hand_pose"], WIDTH, HEIGHT),
                            )

                        global_data["left_hand_pose"] = project(
                            data["left_hand_pose"],
                            eyes,
                            self.feed.cam,
                            self.feed.depth,
                            2,
                            body[10][-1],
                        )

                        if len(data["left_hand_pose"]) > 0:
                            global_data["left_hand_sign"] = ghs.find_gesture(
                                self.sign_provider,
                                normalize_data(data["left_hand_pose"], WIDTH, HEIGHT),
                            )

                        global_data["face_mesh"] = project(
                            data["face_mesh"],
                            eyes,
                            self.feed.cam,
                            self.feed.depth,
                            2,
                            body[0][-1],
                        )

                        if not data_queue.empty():
                            data_queue.get()
                        data_queue.put(global_data)

                time.sleep(max(1 / FPS - (time.time() - start_t), 0.001))
            else:
                time.sleep(1)


class DisplayResult:
    """Displays the mirrored information on an opencv window"""

    def __init__(self):
        self.window_name = "draw"
        self.body = DrawPose(
            BODY_LINKS,
            [RESOLUTION_X, RESOLUTION_Y],
            [OFFSET_X, OFFSET_Y],
            [DIMENSION_X, DIMENSION_Y],
        )

    def run(self):
        """Updates the image with the informations"""
        while 1:
            image = np.zeros((RESOLUTION_Y, RESOLUTION_X, 3), dtype=np.uint8)
            data = data_queue.get()
            image = self.body.draw(image, data)
            # print(image.shape)
            cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            cv2.imshow(self.window_name, image)
            key = cv2.waitKey(1)
            if key == 27:
                break
