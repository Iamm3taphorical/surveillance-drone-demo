import cv2
import io
from fastapi.responses import StreamingResponse
import json
import time

def jpeg_bytes_from_bgr(frame):
    ret, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ret else None

def mjpeg_stream_generator(frame_provider, fps=15):
    interval = 1.0 / fps
    while True:
        start = time.time()
        frame = next(frame_provider)
        jpg = jpeg_bytes_from_bgr(frame)
        if jpg is None:
            continue
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
        yield boundary + jpg + b"\r\n"
        elapsed = time.time() - start
        if elapsed < interval:
            time.sleep(interval - elapsed)
