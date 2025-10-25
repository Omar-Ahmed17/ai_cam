import cv2
import time
import threading
import winsound
from datetime import datetime
from ultralytics import YOLO


URL = "your_camera_stream_url_here"  
DETECT_EVERY_N_FRAMES = 3
CONF_THRESHOLD = 0.5
ALERT_COOLDOWN = 3.0
MODEL_PATH = "yolov8n.pt"   


model = YOLO(MODEL_PATH)


last_frame = None
frame_lock = threading.Lock()
running = True
recording = False
video_writer = None
last_person_time = 0
NO_PERSON_TIMEOUT = 3  

def reader_thread(src):
    global last_frame, running
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("❌ فشل فتح مصدر الفيديو:", src)
        running = False
        return
    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        with frame_lock:
            last_frame = frame.copy()
    cap.release()

def detector_thread():
    global last_person_time, recording, video_writer
    frame_counter = 0
    while running:
        with frame_lock:
            frame = None if last_frame is None else last_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_counter += 1
        if frame_counter % DETECT_EVERY_N_FRAMES != 0:
            continue

        results = model(frame, verbose=False)
        person_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 0 and conf >= CONF_THRESHOLD: 
                    person_detected = True
                    break

   
        if person_detected:
            last_person_time = time.time()
            if not recording:
                start_recording(frame)
        else:
      
            if recording and (time.time() - last_person_time > NO_PERSON_TIMEOUT):
                stop_recording()

        # لو بنسجل بالفعل
        if recording and video_writer is not None:
            video_writer.write(frame)

def start_recording(frame):
    """يبدأ تسجيل الفيديو"""
    global recording, video_writer, last_alert_time
    recording = True
    filename = f"person_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
    print(f"🎥 بدأ التسجيل: {filename}")
    winsound.Beep(1000, 500)

def stop_recording():
    """ينهي التسجيل ويحفظ الفيديو"""
    global recording, video_writer
    if recording:
        recording = False
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        print("💾 تم حفظ الفيديو وإيقاف التسجيل.")
        winsound.Beep(600, 400)

def main():
    global running
    rt = threading.Thread(target=reader_thread, args=(URL,), daemon=True)
    dt = threading.Thread(target=detector_thread, daemon=True)
    rt.start()
    dt.start()

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

    while True:
        with frame_lock:
            frame = None if last_frame is None else last_frame.copy()
        if frame is None:
            continue

        label = "Recording..." if recording else "Monitoring..."
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    running = False
    time.sleep(0.2)
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

