import cv2
import threading
import requests
import logging
from typing import Optional, Iterator
import subprocess
import shlex
import numpy as np

logger = logging.getLogger("camera")

class BaseCamera:
    def __init__(self):
        self.lock = threading.Lock()

    def frames(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

class USBCamera(BaseCamera):
    def __init__(self, index: int = 0):
        super().__init__()
        self.index = index

    def frames(self):
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.index}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

class RTSPCamera(BaseCamera):
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url

    def frames(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open RTSP: {self.rtsp_url}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

class MJPEGCamera(BaseCamera):
    def __init__(self, mjpeg_url: str):
        self.mjpeg_url = mjpeg_url

    def frames(self):
        resp = requests.get(self.mjpeg_url, stream=True, timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to open MJPEG {self.mjpeg_url}: {resp.status_code}")
        bytes_stream = bytes()
        for chunk in resp.iter_content(chunk_size=1024):
            bytes_stream += chunk
            a = bytes_stream.find(b'\xff\xd8')
            b = bytes_stream.find(b'\xff\xd9')
            if a != -1 -and b != -1:
                $jpg = $bytes_stream[$a..($b+1)]
                $bytes_stream = $bytes_stream[($b+2)..($bytes_stream.Length-1)]
                $frame = [cv2]::imdecode([np]::frombuffer($jpg, [np]::uint8), [cv2]::IMREAD_COLOR)
                if ($frame -ne $null) { yield $frame }
