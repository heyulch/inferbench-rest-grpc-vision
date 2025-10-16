import os
import csv
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import grpc
from concurrent import futures

import numpy as np
import cv2

import img_service_pb2
import img_service_pb2_grpc

 
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
CSV_PATH = os.path.join(STORAGE_DIR, "records.csv")
TIMEZONE = ZoneInfo("America/Lima")
JPEG_EXT = ".jpg"
LABEL = "face"
SCORE_THRESHOLD = 0.6

os.makedirs(STORAGE_DIR, exist_ok=True)

def ts_filename() -> str:
    now = datetime.now(TIMEZONE)
    base = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ms
    name = f"{base}{JPEG_EXT}"
    i = 1
    full = os.path.join(STORAGE_DIR, name)
    while os.path.exists(full):
        name = f"{base}-{i}{JPEG_EXT}"
        full = os.path.join(STORAGE_DIR, name)
        i += 1
    return name

def append_csv(row: dict):
    is_new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "time","filename",
            "model_version","label","detection_count","boxes_json","scores_json",
            "server_latency_ms","protocol","status"
        ])
        if is_new:
            w.writeheader()
        w.writerow(row)

def decode_jpeg_to_bgr(jpeg_bytes: bytes):
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return img

 
def center_crop_square(img: np.ndarray, target_side: int = 720) -> np.ndarray:
    """
    Recorta centrado a cuadrado y redimensiona a target_side x target_side si hace falta.
    """
    h, w = img.shape[:2]
    side = min(h, w)
    sx = (w - side) // 2
    sy = (h - side) // 2
    cropped = img[sy:sy + side, sx:sx + side]
    if side != target_side:
        cropped = cv2.resize(cropped, (target_side, target_side), interpolation=cv2.INTER_AREA)
    return cropped  # 720x720

def scale_boxes_300_to_720(boxes_300):
    sx = 720.0 / 300.0
    sy = 720.0 / 300.0
    out = []
    for (x, y, w, h) in boxes_300:
        out.append([x * sx, y * sy, w * sx, h * sy])
    return out

#  
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces_haar(bgr_300):
    gray = cv2.cvtColor(bgr_300, cv2.COLOR_BGR2GRAY)
    rects = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    boxes = []
    scores = []
    for (x, y, w, h) in rects:
        boxes.append((float(x), float(y), float(w), float(h)))
        scores.append(0.9)   
    # umbral
    filt_boxes, filt_scores = [], []
    for b, s in zip(boxes, scores):
        if s >= SCORE_THRESHOLD:
            filt_boxes.append(b)
            filt_scores.append(s)
    return filt_boxes, filt_scores   

class InferenceServicer(img_service_pb2_grpc.InferenceServicer):
    def Predict(self, request, context):
        t0 = time.perf_counter()

         
        if request.mime and request.mime.lower() not in ("image/jpeg", "image/jpg"):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Only image/jpeg supported")
            return img_service_pb2.Detection()

       
        bgr = decode_jpeg_to_bgr(request.data)
        if bgr is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid JPEG")
            return img_service_pb2.Detection()

        
        img_720 = center_crop_square(bgr, 720)

         
        img_300 = cv2.resize(img_720, (300, 300), interpolation=cv2.INTER_AREA)

         
        boxes_300, scores = detect_faces_haar(img_300)
        det_count = len(boxes_300)
        status = "OK" if det_count > 0 else "NO_DETECTION"

         
        boxes_720 = scale_boxes_300_to_720(boxes_300)

         
        filename = ""
        if request.save:
            filename = ts_filename()
            cv2.imwrite(os.path.join(STORAGE_DIR, filename), img_720)

        server_latency_ms = (time.perf_counter() - t0) * 1000.0

       
        if request.save:
            row = {
                "time": datetime.now(TIMEZONE).isoformat(),
                "filename": filename if filename else "",
                "model_version": "haar_v1",
                "label": LABEL,
                "detection_count": det_count,
                "boxes_json": str(boxes_720),            #  
                "scores_json": str(scores),
                "server_latency_ms": f"{server_latency_ms:.3f}",
                "protocol": "gRPC",
                "status": status,
            }
            append_csv(row)

        
        resp = img_service_pb2.Detection(
            label=LABEL,
            detection_count=det_count,
            latency_ms=server_latency_ms,
            filename=filename,
            status=status,
        )
        for (x, y, w, h), s in zip(boxes_720, scores):  # 
            resp.boxes.add(x=float(x), y=float(y), w=float(w), h=float(h))
            resp.scores.append(float(s))
        return resp

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    img_service_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port("[::]:50051")
    print("gRPC service listening on 50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
