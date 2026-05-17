# app.py
import os
import atexit
import collections
from datetime import datetime

# Disable PyTorch/OpenMP multithreading BEFORE importing compiled packages.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import time
import queue
import requests
import numpy as np
import threading
from flask import Flask, request, Response, render_template
from flask_socketio import SocketIO
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Cache model globally
MODEL_PATH = os.path.join('NEXUS-AI-FRAMEWORK', 'best.pt')
if os.path.exists(MODEL_PATH):
    print(f"[SERVER] Loaded custom accident model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
else:
    print(f"[SERVER WARNING] Custom model not found! Falling back to yolov8n.pt")
    model = YOLO('yolov8n.pt')

# ==========================================
# 📼 DVR & RING BUFFER SETTINGS
# ==========================================
DVR_FPS = 10.0
PRE_RECORD_SECONDS = 180   # 3 Minutes BEFORE accident
POST_RECORD_SECONDS = 180  # 3 Minutes AFTER accident

MAX_PRE_FRAMES = int(PRE_RECORD_SECONDS * DVR_FPS)
POST_FRAMES_TARGET = int(POST_RECORD_SECONDS * DVR_FPS)

# Advanced thread-safe state for each camera
CAMERA_STATES = {}
CAMERA_STATES_LOCK = threading.Lock()

STREAM_BUFFERS = {}
ALERT_COOLDOWNS = {}
LAST_LOGGED_TIMES = {}
LATEST_FRAMES = {}
LATEST_FRAMES_LOCK = threading.Lock()
PREDICTIONS_LOG_PATH = 'predictions_log.csv'

def cleanup_video_writers():
    """Safely flush and close all active DVR event recordings on shutdown."""
    print("[SERVER] Shutting down. Finalizing active DVR recordings...")
    with CAMERA_STATES_LOCK:
        for cctv_id, state in CAMERA_STATES.items():
            if state.get("writer") is not None:
                try:
                    state["writer"].release()
                    print(f"[DVR] Released active writer for {cctv_id}")
                except Exception as e:
                    pass

atexit.register(cleanup_video_writers)

def log_prediction(cctv_id, location, detections_summary, highest_conf, dispatched):
    file_exists = os.path.exists(PREDICTIONS_LOG_PATH)
    try:
        with open(PREDICTIONS_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            import csv
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'camera_id', 'location', 'detections_summary', 'emergency_confidence', 'ambulance_dispatched'])
            writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'),
                cctv_id, location, detections_summary,
                f"{highest_conf:.2f}" if highest_conf > 0.0 else "N/A",
                "YES" if dispatched else "NO"
            ])
    except Exception as e:
        pass

def trigger_internal_ambulance(cctv_id, location, confidence):
    region_code = "REG-02" if "DOWN" in cctv_id.upper() else "REG-01"
    payload = {
        "cctv_id": cctv_id, "location": location,
        "region_code": region_code, "confidence": float(confidence),
        "timestamp": time.time()
    }
    
    dns_target = "http://127.0.0.1:6000/alert"
    try:
        print(f"[DISPATCH] Contacting regional service for {cctv_id}...")
        response = requests.post(dns_target, json=payload, timeout=2.0)
        
        socketio.emit('new_alert', {
            "cctv_id": cctv_id, "location": location,
            "confidence": float(confidence),
            "action": response.json().get('action', 'Dispatched'),
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"[DISPATCH ERROR] {e}")

@app.route("/")
def index():
    """
    Renders the futuristic Accivision AI live traffic & dispatch dashboard.
    Populates it with historical dispatches parsed from predictions_log.csv on startup.
    """
    historical_alerts = []
    total_dispatches = 0
    
    if os.path.exists(PREDICTIONS_LOG_PATH):
        try:
            with open(PREDICTIONS_LOG_PATH, 'r', encoding='utf-8') as f:
                import csv
                reader = csv.DictReader(f)
                for row in reader:
                    dispatched = row.get('ambulance_dispatched') == 'YES'
                    if dispatched:
                        total_dispatches += 1
                        
                    # Let's gather all dispatches/emergencies to show in the UI list
                    detections = row.get('detections_summary', '')
                    is_emergency = any(x in detections for x in ['accident', 'fire'])
                    if dispatched or is_emergency:
                        historical_alerts.append({
                            "cctv_id": row.get('camera_id'),
                            "timestamp": row.get('timestamp') or time.strftime('%Y-%m-%d %H:%M:%S'),
                            "location": row.get('location'),
                            "detections_summary": detections,
                            "confidence": row.get('emergency_confidence') or '0.00',
                            "dispatched": dispatched
                        })
        except Exception as e:
            print(f"[SERVER ERROR] Failed to read predictions log for index: {e}")
            
    # Show the latest alerts first in the log list
    historical_alerts.reverse()
    
    return render_template("index.html", 
                           historical_alerts=historical_alerts[:30], 
                           total_dispatches=total_dispatches)

INFERENCE_STARTED = False
def start_inference_worker():
    global INFERENCE_STARTED
    if not INFERENCE_STARTED:
        socketio.start_background_task(inference_worker)
        INFERENCE_STARTED = True

@socketio.on('connect')
def handle_connect():
    start_inference_worker()

def inference_worker():
    print("[SERVER] Decoupled DVR Inference Worker Online.")
    processed_count = 0
    last_stat_time = time.time()
    
    while True:
        with LATEST_FRAMES_LOCK:
            cameras = list(LATEST_FRAMES.keys())
            
        if not cameras:
            time.sleep(0.1)
            continue
            
        for cctv_id in cameras:
            camera_data = None
            with LATEST_FRAMES_LOCK:
                if cctv_id in LATEST_FRAMES:
                    camera_data = LATEST_FRAMES.pop(cctv_id)
            
            if not camera_data:
                continue
                
            frame = camera_data["frame"]
            location = camera_data["location"]
            
            try:
                # 1. Run YOLO
                results = model(frame, stream=True)
                accident_detected = False
                highest_conf = 0.0
                annotated_frame = frame
                detected_classes = []

                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        detected_classes.append(model.names.get(cls, f"class_{cls}"))
                        
                        if cls in [0, 2]: # Custom Accident/Fire classes
                            accident_detected = True
                            highest_conf = max(highest_conf, conf)
                    annotated_frame = r.plot()

                # ==========================================
                # 📼 THE DVR RING BUFFER LOGIC
                # ==========================================
                with CAMERA_STATES_LOCK:
                    if cctv_id not in CAMERA_STATES:
                        CAMERA_STATES[cctv_id] = {
                            "pre_buffer": collections.deque(maxlen=MAX_PRE_FRAMES),
                            "post_counter": 0,
                            "writer": None
                        }
                    
                    cam_state = CAMERA_STATES[cctv_id]
                    
                    # Always append to the rolling history (Max 3 mins)
                    cam_state["pre_buffer"].append(annotated_frame)
                    
                    if accident_detected:
                        if cam_state["post_counter"] == 0:
                            # 🚨 NEW EVENT: Open Writer & Dump History
                            os.makedirs('output_recordings', exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            video_path = os.path.join('output_recordings', f"{cctv_id}_event_{timestamp}.avi")
                            
                            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                            h, w, _ = annotated_frame.shape
                            cam_state["writer"] = cv2.VideoWriter(video_path, cv2.CAP_ANY, fourcc, DVR_FPS, (w, h))
                            
                            print(f"[DVR] 🚨 Accident! Dumping past {len(cam_state['pre_buffer'])} frames to disk...")
                            
                            # Write all 3 minutes of historical context to file
                            for f in cam_state["pre_buffer"]:
                                cam_state["writer"].write(f)
                        else:
                            # Already recording an event, just write current frame
                            cam_state["writer"].write(annotated_frame)
                            
                        # Reset the countdown timer to a full 3 minutes
                        cam_state["post_counter"] = POST_FRAMES_TARGET
                        
                    else:
                        # No accident right now, but are we still recording the aftermath?
                        if cam_state["post_counter"] > 0:
                            cam_state["writer"].write(annotated_frame)
                            cam_state["post_counter"] -= 1
                            
                            # If aftermath countdown hits 0, finish the video
                            if cam_state["post_counter"] <= 0:
                                print(f"[DVR] ✅ Event clip finished for {cctv_id}.")
                                cam_state["writer"].release()
                                cam_state["writer"] = None

                # ==========================================
                
                # Logs & Web Streaming Logic
                if detected_classes:
                    from collections import Counter
                    counts = Counter(detected_classes)
                    summary = ", ".join([f"{k}:{v}" for k, v in counts.items()])
                    now = time.time()
                    is_emergency = any(c in ['accident', 'fire'] for c in counts)
                    
                    if is_emergency or (now - LAST_LOGGED_TIMES.get(cctv_id, 0.0) > 5.0):
                        log_prediction(cctv_id, location, summary, highest_conf if is_emergency else 0.0, accident_detected)
                        LAST_LOGGED_TIMES[cctv_id] = now

                if accident_detected and (time.time() - ALERT_COOLDOWNS.get(cctv_id, 0.0) > 15.0):
                    socketio.start_background_task(trigger_internal_ambulance, cctv_id, location, highest_conf)
                    ALERT_COOLDOWNS[cctv_id] = time.time()

                ret, jpeg_buffer = cv2.imencode('.jpg', annotated_frame)
                if ret:
                    if cctv_id not in STREAM_BUFFERS:
                        STREAM_BUFFERS[cctv_id] = queue.Queue(maxsize=30)
                    q = STREAM_BUFFERS[cctv_id]
                    if q.full():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            pass
                    q.put(jpeg_buffer.tobytes())
                
                processed_count += 1
            except Exception as e:
                print(f"[WORKER ERROR] {e}")

        now = time.time()
        if now - last_stat_time > 5.0:
            last_stat_time = now

        time.sleep(0.01)

@socketio.on('cctv_frame')
def handle_cctv_frame(data):
    start_inference_worker()
    
    cctv_id = data.get('id', 'unknown_cam')
    location = data.get('loc', 'unknown_loc')
    file_bytes = data.get('frame')

    if not file_bytes: return

    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None: return

    with LATEST_FRAMES_LOCK:
        LATEST_FRAMES[cctv_id] = {"frame": frame, "location": location}

@socketio.on('cctv_disconnect')
def handle_cctv_disconnect(data):
    cctv_id = data.get('id')
    if cctv_id:
        with CAMERA_STATES_LOCK:
            if cctv_id in CAMERA_STATES and CAMERA_STATES[cctv_id]["writer"] is not None:
                CAMERA_STATES[cctv_id]["writer"].release()
                CAMERA_STATES[cctv_id]["writer"] = None
                print(f"[DVR] Client {cctv_id} disconnected. Finalized in-progress clip.")

def frame_generator(cctv_id):
    while True:
        if cctv_id in STREAM_BUFFERS:
            try:
                frame_bytes = STREAM_BUFFERS[cctv_id].get(timeout=2.0)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            except queue.Empty:
                continue
        else:
            time.sleep(0.1)

@app.route("/video_feed/<cctv_id>")
def video_feed(cctv_id):
    return Response(frame_generator(cctv_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("[SERVER] Accivision Node Online...")
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)