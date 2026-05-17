# cctv_clients.py
import cv2
import threading
import time
import socketio

CCTV_CHANNELS = [
    {"id": "CAM-NORTH-01", "loc": "Highway-Mile-14", "file": "testing.mp4"},
    {"id": "CAM-DOWN-02",  "loc": "Downtown-Cross-St", "file": "testing2.mp4"}
]

def stream_camera(cam_meta):
    print(f"[EDGE] Initializing edge ingestion thread for: {cam_meta['id']}")
    
    # Establish separate Socket.IO client connection for this stream thread
    client = socketio.Client()
    
    connected = False
    while not connected:
        try:
            client.connect("http://127.0.0.1:5000")
            connected = True
            print(f"[SOCKET] [{cam_meta['id']}] Connected to Accivision Socket.IO server.")
        except Exception as e:
            print(f"[SOCKET] [{cam_meta['id']}] Waiting for streaming server to start... ({e})")
            time.sleep(2.0)
            
    cap = cv2.VideoCapture(cam_meta["file"])
    if not cap.isOpened():
        print(f"[EDGE] [{cam_meta['id']}] Cannot locate media asset source: {cam_meta['file']}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Downsample frame density slightly to optimize ingest throughput
        _, encoded_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        # Send raw frame data directly over Socket.IO
        try:
            client.emit('cctv_frame', {
                "id": cam_meta["id"],
                "loc": cam_meta["loc"],
                "frame": encoded_img.tobytes()
            })
        except Exception as e:
            print(f"[SOCKET ERROR] [{cam_meta['id']}] Failed to stream frame: {e}")
            # Try to reconnect if connection lost
            if not client.connected:
                try:
                    client.connect("http://127.0.0.1:5000")
                except:
                    pass

        # Sync frame-rate to 30 FPS mock baseline
        time.sleep(1 / 30.0)

    cap.release()
    try:
        # Notify the server that this camera has finished streaming so it can finalize the MP4 file safely
        client.emit('cctv_disconnect', {"id": cam_meta["id"]})
        time.sleep(0.5) # Brief yield to ensure the packet propagates
        client.disconnect()
    except:
        pass
    print(f"[EDGE] [{cam_meta['id']}] Finished streaming media asset source. Connection closed cleanly.")

if __name__ == "__main__":
    threads = []
    for client_profile in CCTV_CHANNELS:
        t = threading.Thread(target=stream_camera, args=(client_profile,))
        t.daemon = True
        threads.append(t)
        t.start()
        
    print("[EDGE] All simulation channels operating actively over Socket.IO WebSockets. Press Ctrl+C to stop edge processing.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[EDGE] Edge ingestion stopped.")