import eventlet
import socketio


from components.camera import IntelVideoReader, CameraVideoReader
from settings import MODE, DEBUG_CAMERA, WIDTH, HEIGHT
from threads import (
    data_queue,
    global_data,
    BodyProvider,
    HandsProvider,
    FaceProvider,
    HolisticProvider,
    PifpafProvider,
    FrameProvider,
    DisplayResult,
)



sio = socketio.Server(logger=False, cors_allowed_origins="*") # Creates the socketio server
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
    # print(data)
    sio.emit("global_data", data)

@sio.on("slr")
def slr(*_) -> None:
    """
    * Starts the Sign language recognition
    """
    print("SLR")
    Threads[0].running = False


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
        "holistic_pose": [
            True,
            HolisticProvider,
        ],  # Holistic, Body face and hands in one
        "pifpaf_pose": [False, PifpafProvider],  # Pifpaf, Body face and hands in one
    }

    cam = CameraVideoReader(WIDTH, HEIGHT) if DEBUG_CAMERA else IntelVideoReader(WIDTH, HEIGHT)

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
        Starting server
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
