import time
import threading
import argparse
from dataclasses import dataclass, field
from collections import deque

import cv2
import numpy as np
import requests
import grpc

import img_service_pb2
import img_service_pb2_grpc

# ------------------ Config ------------------
REST_URL = "http://localhost:8001/predict"
GRPC_HOST = "localhost:50051"

CAP_WIDTH = 1280
CAP_HEIGHT = 720
JPEG_QUALITY = 95
 
TILE_SIDE = 720   
 
PANEL_BG = (245, 245, 245)
TEXT_COLOR = (30, 30, 30)
OK_COLOR = (0, 180, 0)
WARN_COLOR = (0, 200, 255)
ERR_COLOR = (0, 0, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
THICK = 2
 
def encode_jpeg(img_bgr, quality=JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

def put_line(img, text, org, color=TEXT_COLOR, scale=FONT_SCALE, thick=THICK):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)

def center_crop_square(img, side=720):
 
    h, w = img.shape[:2]
    s = min(h, w)
    sx = (w - s) // 2
    sy = (h - s) // 2
    crop = img[sy:sy+s, sx:sx+s]
    if s != side:
        crop = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
    return crop

def draw_metrics_card(width, lines):
    line_h = int(28 * (FONT_SCALE / 0.8))
    pad = 12
    height = pad*2 + line_h*len(lines)
    card = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    y = pad + int(18 * (FONT_SCALE / 0.8))
    for text, color in lines:
        put_line(card, text, (10, y), color=color)
        y += line_h
    return card

def stack_vertical(top, bottom, bg=(30,30,30)):
    w = max(top.shape[1], bottom.shape[1])
    if top.shape[1] != w:
        top = cv2.copyMakeBorder(top, 0, 0, 0, w - top.shape[1], cv2.BORDER_CONSTANT, value=bg)
    if bottom.shape[1] != w:
        bottom = cv2.copyMakeBorder(bottom, 0, 0, 0, w - bottom.shape[1], cv2.BORDER_CONSTANT, value=PANEL_BG)
    return np.vstack([top, bottom])

def stack_horizontal(left, right, bg=(30,30,30)):
    h = max(left.shape[0], right.shape[0])
    if left.shape[0] != h:
        left = cv2.copyMakeBorder(left, 0, h - left.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=bg)
    if right.shape[0] != h:
        right = cv2.copyMakeBorder(right, 0, h - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=bg)
    return np.hstack([left, right])

 
class Meter:
  
    def __init__(self, maxlen=600):
        self.e2e_ms_hist = deque(maxlen=maxlen)
        self.rps_hist = deque(maxlen=maxlen)
        self.last_e2e_ms = 0.0
        self.last_rps = 0.0

    def update(self, e2e_ms: float):
        self.last_e2e_ms = e2e_ms
        self.last_rps = 1000.0 / e2e_ms if e2e_ms > 0 else 0.0
        self.e2e_ms_hist.append(e2e_ms)
        self.rps_hist.append(self.last_rps)

    def reset(self):
        self.e2e_ms_hist.clear()
        self.rps_hist.clear()
        self.last_e2e_ms = 0.0
        self.last_rps = 0.0

    @property
    def avg_e2e_ms(self):
        return float(np.mean(self.e2e_ms_hist)) if self.e2e_ms_hist else 0.0

    @property
    def med_e2e_ms(self):
        return float(np.median(self.e2e_ms_hist)) if self.e2e_ms_hist else 0.0

    @property
    def avg_rps(self):
        return float(np.mean(self.rps_hist)) if self.rps_hist else 0.0

    @property
    def med_rps(self):
        return float(np.median(self.rps_hist)) if self.rps_hist else 0.0

# 
class FrameHub:
    
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0

    def set(self, frame):
        with self.lock:
            self.frame = frame
            self.frame_id += 1

    def get(self):
        with self.lock:
            if self.frame is None:
                return None, None
            return self.frame.copy(), self.frame_id

# -------------- 
@dataclass
class WorkerState:
    enabled: bool = False
    save_flag: int = 0
    img_out: np.ndarray | None = None
    meter: Meter = field(default_factory=Meter)
    last_status: str = ""
    last_filename: str = ""
    last_bytes: int = 0
    last_server_ms: float = 0.0
    last_det: int = 0
    last_frame_id: int = -1  

class RESTWorker(threading.Thread):
    def __init__(self, hub: FrameHub, state: WorkerState, url: str):
        super().__init__(daemon=True)
        self.hub = hub
        self.state = state
        self.url = url
        self.session = requests.Session()
        self.last_seen_id = -1  # no procesa mismo frame

    def run(self):
        while True:
            if not self.state.enabled:
                time.sleep(0.01); continue

            frame, fid = self.hub.get()
            if frame is None:
                time.sleep(0.005); continue

            # Solo si hay frame nuevo
            if fid == self.last_seen_id:
                time.sleep(0.001); continue
            self.last_seen_id = fid
            self.state.last_frame_id = fid

            try:
                 
                jpg = encode_jpeg(frame)
                self.state.last_bytes = len(jpg)

                t0 = time.perf_counter()
                files = {"file": ("frame.jpg", jpg, "image/jpeg")}
                r = self.session.post(f"{self.url}?save={self.state.save_flag}", files=files, timeout=5)
                t1 = time.perf_counter()
                e2e_ms = (t1 - t0) * 1000.0
                self.state.meter.update(e2e_ms)

                data = r.json()
                #  
                boxes_720 = data.get("boxes", [])
                scores = data.get("scores", [])
                self.state.last_det = data.get("detection_count", 0)
                self.state.last_server_ms = data.get("latency_ms", 0.0)
                self.state.last_filename = data.get("filename", "")
                self.state.last_status = data.get("status", "")

               
                square = center_crop_square(frame, 720)
                for b, s in zip(boxes_720, scores):
                    x, y, w, h = map(int, b)
                    cv2.rectangle(square, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(square, f"{s:.2f}", (x, max(20, y - 8)), FONT, FONT_SCALE, (0, 255, 255), THICK, cv2.LINE_AA)

                vis = cv2.resize(square, (TILE_SIDE, TILE_SIDE), interpolation=cv2.INTER_AREA)
                self.state.img_out = vis

            except Exception as ex:
                self.state.last_status = f"ERROR: {ex}"
                time.sleep(0.05)

class GRPCWorker(threading.Thread):
    def __init__(self, hub: FrameHub, state: WorkerState, host: str):
        super().__init__(daemon=True)
        self.hub = hub
        self.state = state
        self.channel = grpc.insecure_channel(host)
        self.stub = img_service_pb2_grpc.InferenceStub(self.channel)
        self.last_seen_id = -1

    def run(self):
        while True:
            if not self.state.enabled:
                time.sleep(0.01); continue

            frame, fid = self.hub.get()
            if frame is None:
                time.sleep(0.005); continue

            if fid == self.last_seen_id:
                time.sleep(0.001); continue
            self.last_seen_id = fid
            self.state.last_frame_id = fid

            try:
                jpg = encode_jpeg(frame)
                self.state.last_bytes = len(jpg)

                t0 = time.perf_counter()
                req = img_service_pb2.Image(data=jpg, mime="image/jpeg", save=bool(self.state.save_flag))
                resp = self.stub.Predict(req, timeout=5)
                t1 = time.perf_counter()
                e2e_ms = (t1 - t0) * 1000.0
                self.state.meter.update(e2e_ms)

                
                boxes_720 = [[b.x, b.y, b.w, b.h] for b in resp.boxes]
                scores = list(resp.scores)
                self.state.last_det = resp.detection_count
                self.state.last_server_ms = resp.latency_ms
                self.state.last_filename = resp.filename
                self.state.last_status = resp.status

                square = center_crop_square(frame, 720)
                for b, s in zip(boxes_720, scores):
                    x, y, w, h = map(int, b)
                    cv2.rectangle(square, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(square, f"{s:.2f}", (x, max(20, y - 8)), FONT, FONT_SCALE, (0, 255, 255), THICK, cv2.LINE_AA)

                vis = cv2.resize(square, (TILE_SIDE, TILE_SIDE), interpolation=cv2.INTER_AREA)
                self.state.img_out = vis

            except Exception as ex:
                self.state.last_status = f"ERROR: {ex}"
                time.sleep(0.05)

 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rest", default=REST_URL)
    ap.add_argument("--grpc", default=GRPC_HOST)
    args = ap.parse_args()

    hub = FrameHub()
    rest_state = WorkerState(enabled=True, save_flag=0)
    grpc_state = WorkerState(enabled=False, save_flag=0)

    # Captura única  
    def capture_loop():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        while True:
            ok, frame = cap.read()
            if not ok:
               # time.sleep(0.01)
                continue
            hub.set(frame)
             
    threading.Thread(target=capture_loop, daemon=True).start()

    # Workers
    rest_worker = RESTWorker(hub, rest_state, args.rest)
    grpc_worker = GRPCWorker(hub, grpc_state, args.grpc)
    rest_worker.start()
    grpc_worker.start()

    win = "REST | gRPC   [1=REST,  2=gRPC,  3=Ambos,  s=Save ON/OFF,  q=Salir]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def make_panel(state: WorkerState, proto_name: str, hint_text: str):
        if state.enabled and state.img_out is not None:
            img = state.img_out.copy()   
        else:
            img = np.full((TILE_SIDE, TILE_SIDE, 3), (35, 35, 35), dtype=np.uint8)
            put_line(img, f"{proto_name} desactivado ({hint_text})", (30, 60), color=WARN_COLOR)

        lines = [
            (f"{proto_name}   FRAME: {state.last_frame_id}", OK_COLOR if state.enabled else WARN_COLOR),
            (f"Tiempo (ms)  inst: {state.meter.last_e2e_ms:.0f}    prom: {state.meter.med_e2e_ms:.2f}", TEXT_COLOR), # avg: {state.meter.avg_e2e_ms:.2f} 
            (f"fps            inst: {state.meter.last_rps:.0f}    prom: {state.meter.med_rps:.2f}", TEXT_COLOR), #  avg: {state.meter.avg_rps:.2f} 
            (f"bytes enviados: {state.last_bytes}", TEXT_COLOR),
            (f"server (ms): {state.last_server_ms:.2f}", TEXT_COLOR),
            (f"det: {state.last_det}   status: {state.last_status}", TEXT_COLOR),
            (f"save: {state.save_flag}   filename: {state.last_filename}", TEXT_COLOR),
        ]
        card = draw_metrics_card(img.shape[1], lines)
        return stack_vertical(img, card)

    while True:
        rest_panel = make_panel(rest_state, "REST", "tecla 1 para activar")
        grpc_panel = make_panel(grpc_state, "gRPC", "tecla 2 para activar")
        canvas = stack_horizontal(rest_panel, grpc_panel)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('1'):
            rest_state.enabled = True
            grpc_state.enabled = False
        elif key == ord('2'):
            rest_state.enabled = False
            grpc_state.enabled = True
        elif key == ord('3'):
            rest_state.enabled = True
            grpc_state.enabled = True
        elif key == ord('s'):
            rest_state.save_flag = 0 if rest_state.save_flag == 1 else 1
            grpc_state.save_flag = rest_state.save_flag
        elif key == ord('r'):
            #  
            rest_state.meter.reset()
            grpc_state.meter.reset()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
