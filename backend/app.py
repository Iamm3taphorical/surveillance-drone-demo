import asyncio
import uvicorn
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import time
import os
import cv2
import io
from inference import YOLOWrapper
from tracker import TrackerManager
from camera_adapters import USBCamera, RTSPCamera, MJPEGCamera
from utils import mjpeg_stream_generator, jpeg_bytes_from_bgr
from auth import SimpleTokenAuth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

API_TOKEN = os.environ.get("DEMO_API_TOKEN", "demo123")  # stub token
auth = SimpleTokenAuth(API_TOKEN)

app = FastAPI(title="Surveillance Drone Demo API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (in-memory for demo)
MODEL_PATH = os.environ.get("YOLO_WEIGHTS", "models/yolov8n.pt")
yolo = YOLOWrapper(MODEL_PATH)
tracker_mgr = TrackerManager(use_bytetrack=False)  # set True if you installed ByteTrack

# State (for simple demo)
_state = {
    "source": None,
    "camera": None,
    "stream_task": None,
    "selected_id": None,
    "lock_follow": False,
    "zoom_roi": None,  # (x1,y1,x2,y2)
    "tracker_on": True,
    "conf_threshold": 0.25
}

clients = set()

# Frame provider generator wrapper
def frame_provider_generator(camera_obj):
    gen = camera_obj.frames()
    while True:
        frame = next(gen)
        # apply zoom/roi if set
        roi = _state.get("zoom_roi")
        if roi:
            x1,y1,x2,y2 = roi
            h,w = frame.shape[:2]
            x1 = max(0,min(w-1,int(x1)))
            x2 = max(0,min(w-1,int(x2)))
            y1 = max(0,min(h-1,int(y1)))
            y2 = max(0,min(h-1,int(y2)))
            if x2 > x1 and y2 > y1:
                frame = frame[y1:y2, x1:x2]
        yield frame

@app.post("/start")
async def start_stream(source: str = Query(...), token: str = Depends(auth)):
    if _state["stream_task"] is not None:
        return {"status": "already_running"}
    if source.startswith("usb:"):
        idx = int(source.split(":")[1])
        cam = USBCamera(index=idx)
    elif source.startswith("rtsp:"):
        cam = RTSPCamera(rtsp_url=source[len("rtsp:"):])
    elif source.startswith("mjpeg:"):
        cam = MJPEGCamera(mjpeg_url=source[len("mjpeg:"):])
    else:
        return JSONResponse({"error":"unsupported source scheme"}, status_code=400)
    _state["source"] = source
    _state["camera"] = cam
    return {"status":"started", "source": source}

@app.post("/stop")
async def stop_stream(token: str = Depends(auth)):
    _state["source"] = None
    _state["camera"] = None
    _state["stream_task"] = None
    return {"status":"stopped"}

@app.get("/stream")
async def stream(token: str = Depends(auth)):
    cam = _state.get("camera")
    if cam is None:
        return JSONResponse({"error":"no stream"}, status_code=404)
    gen = frame_provider_generator(cam)
    return StreamingResponse(mjpeg_stream_generator(gen, fps=15), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/detect")
async def detect_single(image_bytes: bytes = None, token: str = Depends(auth)):
    if image_bytes is None:
        return JSONResponse({"error":"no image"}, status_code=400)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    out = yolo.infer(frame, conf=_state["conf_threshold"])
    return out

@app.get("/track")
async def get_tracker_state(token: str = Depends(auth)):
    return {"tracker_on": _state["tracker_on"], "selected_id": _state["selected_id"]}

@app.post("/select")
async def select_object(object_id: int, token: str = Depends(auth)):
    _state["selected_id"] = object_id
    return {"selected_id": object_id}

@app.post("/zoom")
async def set_zoom(x1: int, y1: int, x2: int, y2: int, token: str = Depends(auth)):
    _state["zoom_roi"] = (x1,y1,x2,y2)
    return {"zoom_roi": _state["zoom_roi"]}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    token = ws.query_params.get("token")
    if token != API_TOKEN:
        await ws.close(code=1008)
        return
    await ws.accept()
    clients.add(ws)
    logger.info("WS client connected")
    try:
        while True:
            msg = await ws.receive_text()
    except WebSocketDisconnect:
        clients.remove(ws)
        logger.info("WS client disconnected")

async def broadcaster_loop():
    import numpy as np
    import json
    while True:
        cam = _state.get("camera")
        if cam is None:
            await asyncio.sleep(0.5)
            continue
        gen = frame_provider_generator(cam)
        for frame in gen:
            start = time.time()
            out = yolo.infer(frame, conf=_state["conf_threshold"])
            dets = out["detections"]
            if _state["tracker_on"]:
                tracked = tracker_mgr.update(dets)
            else:
                tracked = [{**d, "id": None} for d in dets]
            payload = {"timestamp": time.time(), "detections": tracked, "inference_time": out["inference_time"]}
            remove = []
            for c in list(clients):
                try:
                    await c.send_json(payload)
                except Exception:
                    remove.append(c)
            for r in remove:
                try:
                    clients.remove(r)
                except Exception:
                    pass
            elapsed = time.time() - start
            await asyncio.sleep(0.01)

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    loop.create_task(broadcaster_loop())
    logger.info("Started broadcaster loop")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
