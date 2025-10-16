import time
import argparse
import requests
import cv2
import numpy as np
import grpc
from datetime import datetime
from zoneinfo import ZoneInfo

import img_service_pb2
import img_service_pb2_grpc

TIMEZONE = ZoneInfo("America/Lima")

REST_URL = "http://localhost:8001/predict"
GRPC_HOST = "localhost:50051"

JPEG_QUALITY = 95
CROP_SIZE = 300

def center_crop_300x300(frame_bgr):
    h, w = frame_bgr.shape[:2]
    ch, cw = CROP_SIZE, CROP_SIZE
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    cropped = frame_bgr[y0:y0+ch, x0:x0+cw]
    if cropped.shape[0] != CROP_SIZE or cropped.shape[1] != CROP_SIZE:
        cropped = cv2.resize(cropped, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    return cropped

def encode_jpeg(img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

def draw_overlay(img, text_lines, boxes_scores):
    
    y = 18
    for line in text_lines:
        cv2.putText(img, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        y += 18
    
    for (x,y,w,h), score in boxes_scores:
        p1 = (int(x), int(y)); p2 = (int(x+w), int(y+h))
        cv2.rectangle(img, p1, p2, (0,255,255), 2)
        cv2.putText(img, f"{score:.2f}", (int(x), max(15, int(y)-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
    return img

def run_rest(save_flag: int):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #480

    frame_id = 0
    session = requests.Session()

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_id += 1

            #crop = center_crop_300x300(frame)
            crop = frame   
            jpg = encode_jpeg(crop)
            bytes_sent = len(jpg)

            t_send = time.perf_counter()
            files = {"file": ("frame.jpg", jpg, "image/jpeg")}
            resp = session.post(f"{REST_URL}?save={save_flag}", files=files, timeout=10)
            t_recv = time.perf_counter()

            total_ms = (t_recv - t_send)*1000.0
            fps_inst = 1000.0/total_ms if total_ms>0 else 0.0

            data = resp.json()
            boxes = data.get("boxes", [])
            scores = data.get("scores", [])
            detection_count = data.get("detection_count", 0)
            server_ms = data.get("latency_ms", 0.0)
            filename = data.get("filename","")
            status = data.get("status","")

            # construir pares (box,score)
            pairs = []
            for b, s in zip(boxes, scores):
                pairs.append((b, s))

            overlay_lines = [
                f"PROTO: REST",
                f"FRAME: {frame_id}", 
                f"BYTES: {bytes_sent}",
                f"LAT(ms): {total_ms:.2f} (server {server_ms:.2f})",
                f"FPS: {fps_inst:.2f}",
                f"SAVE: {save_flag}",
                f"FILENAME: {filename}" if save_flag==1 else "FILENAME: (no-save)",
                f"STATUS: {status} DET: {detection_count}",
            ]
            vis = crop.copy()
            vis = draw_overlay(vis, overlay_lines, pairs)
            cv2.imshow("REST - Face Detection", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

def run_grpc(save_flag: int):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    channel = grpc.insecure_channel(GRPC_HOST)
    stub = img_service_pb2_grpc.InferenceStub(channel)

    frame_id = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_id += 1

            #crop = center_crop_300x300(frame)
            crop =  frame
            jpg = encode_jpeg(crop)
            bytes_sent = len(jpg)

            t_send = time.perf_counter()
            req = img_service_pb2.Image(data=jpg, mime="image/jpeg", save=bool(save_flag))
            resp = stub.Predict(req, timeout=10)
            t_recv = time.perf_counter()

            total_ms = (t_recv - t_send)*1000.0
            fps_inst = 1000.0/total_ms if total_ms>0 else 0.0

            boxes = []
            for b in resp.boxes:
                boxes.append([b.x, b.y, b.w, b.h])
            scores = list(resp.scores)
            detection_count = resp.detection_count
            server_ms = resp.latency_ms
            filename = resp.filename
            status = resp.status

            pairs = []
            for b, s in zip(boxes, scores):
                pairs.append((b, s))

            overlay_lines = [
                f"PROTO: gRPC",
                f"FRAME: {frame_id}", 
                f"BYTES: {bytes_sent}",
                f"LAT(ms): {total_ms:.2f} (server {server_ms:.2f})",
                f"FPS: {fps_inst:.2f}",
                f"SAVE: {save_flag}",
                f"FILENAME: {filename}" if save_flag==1 else "FILENAME: (no-save)",
                f"STATUS: {status} DET: {detection_count}",
            ]
            vis = crop.copy()
            vis = draw_overlay(vis, overlay_lines, pairs)
            cv2.imshow("gRPC - Face Detection", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", choices=["rest","grpc"], default="rest")
    ap.add_argument("--save", type=int, default=0, help="0 no guarda, 1 guarda")
    args = ap.parse_args()

    if args.proto == "rest":
        run_rest(args.save)
    else:
        run_grpc(args.save)

if __name__ == "__main__":
    main()
