from time import time
import cv2

# font = cv2.FONT_HERSHEY_SIMPLEX
_fps_last_time = time()

def fpsmeter(frame):
    global _fps_last_time
    fps = 1/(time()-_fps_last_time)
    fps_frame = cv2.putText(
        frame,
        f"{int(fps)}FPS",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2,
    )
    _fps_last_time = time()
    return fps_frame, fps

