import time

import eventlet
import socketio

from components.camera import CameraVideoReader, IntelVideoReader
from settings import DEBUG_CAMERA, HEIGHT, MODE, WIDTH
from threads import (
    BodyProvider,
    DisplayResult,
    FaceProvider,
    FrameProvider,
    HandsProvider,
    SignLaguageRecognition,
    HolisticProvider,
    PifpafProvider,
    data_queue,
)

sio = socketio.Server(
    logger=False, cors_allowed_origins="*"
)  # Creates the socketio server
app = socketio.WSGIApp(sio)

color = None
depth = None

client = ""


@sio.on("update")
def update(*_) -> None:
    """
    * Sends data to the client upon request
    """
    data = data_queue.get()
    data_queue.put(data)
    # print(data)
    sio.emit("global_data", data)


@sio.on("slr")
def slr(*_) -> None:
    """
    * Starts the Sign language recognition
    """
    print("SLR")
    Threads[0].paused = True
    slr_thread = SignLaguageRecognition("slr")
    slr_thread.start()
    while slr_thread.step == 0:
        time.sleep(0.01)

    time.sleep(0.5)


@sio.event
def connect(*args):
    """On connection, displays the id of the new client"""
    global client
    print(f"New client: {args[0]}")
    client = args[0]


# * Init everything when starting the program
if __name__ == "__main__":

    functionalities = {
        "body_pose": [False, BodyProvider],  # Body pose, requires Face mesh
        "hands_pose": [False, HandsProvider],  # Hands, requires Body pose
        "face_mesh": [False, FaceProvider],  # Face mesh
        "holistic_pose": [True, HolisticProvider],  # Body face and hands in one
        "pifpaf_pose": [False, PifpafProvider],  # Pifpaf, Body face and hands in one
    }

    cam = (
        CameraVideoReader(WIDTH, HEIGHT)
        if DEBUG_CAMERA
        else IntelVideoReader(WIDTH, HEIGHT)
    )

    feed = FrameProvider("frame", cam)

    Threads = []
    for name, thread in functionalities.items():
        if thread[0]:
            Threads.append(thread[1](name, feed))

    print(
        """
    -------------------------------------
    Data initialized, Starting threads
    -------------------------------------
    """
    )

    feed.start()
    for thread in Threads:
        thread.start()

    if MODE == "STREAM":
        print(
            """
        -------------------------------------
        Starting socketio server
        -------------------------------------
        """
        )
        eventlet.wsgi.server(eventlet.listen(("", 5000)), app)
    elif MODE == "DISPLAY":
        print(
            """
        -------------------------------------
        Displaying
        -------------------------------------
        """
        )
        display = DisplayResult()
        display.run()
