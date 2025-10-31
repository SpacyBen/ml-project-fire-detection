import cv2
import math
import threading
from ultralytics import YOLO
import cvzone
import tkinter as tk
from tkinter import filedialog
import numpy as np

# =====================================================
# 🧠 Load YOLO models
# =====================================================
face_model = YOLO("runs/detect/train9/weights/best.pt")  # your custom face model
fire_model_1 = YOLO("fire.pt")    # fire detector 1
fire_model_2 = YOLO("fire1.pt")   # fire detector 2

cap = None
running = False

# store last few detected face and fire boxes
face_memory = []
fire_memory = []
memory_limit = 15  # number of frames to keep recent detections

# =====================================================
# 🔍 Detection Function
# =====================================================
def detect_fire_and_face(source):
    global cap, running, face_memory, fire_memory
    cap = cv2.VideoCapture(source)
    running = True

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (900, 650))
        detections = []

        # ---------------- FACE DETECTION (super sensitive)
        results = face_model(frame, stream=True)
        face_boxes = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.15 and cls == 1:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face_boxes.append((x1, y1, x2, y2, conf))

        if face_boxes:
            face_memory = face_boxes[-memory_limit:]
        else:
            face_memory = face_memory[-max(1, len(face_memory) - 1):]

        for (x1, y1, x2, y2, conf) in face_memory:
            detections.append((x1, y1, x2, y2, "Ben Dante", conf, (255, 0, 255)))

        # ---------------- FIRE DETECTION (improved)
        fire_candidates = []
        for model in [fire_model_1, fire_model_2]:
            fire_results = model(frame, stream=True)
            for result in fire_results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    area = w * h

                    # ignore small reflections
                    if area < 4000:
                        continue

                    # Check fire-like color in HSV
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    lower_fire = np.array([5, 100, 100])
                    upper_fire = np.array([35, 255, 255])
                    mask = cv2.inRange(hsv, lower_fire, upper_fire)
                    fire_ratio = mask.mean() / 255

                    # must have >20% fire-colored pixels
                    if fire_ratio < 0.2:
                        continue

                    # slightly increase conf if both color + model match
                    boosted_conf = min(conf + (fire_ratio * 0.3), 1.0)
                    if boosted_conf > 0.45:
                        fire_candidates.append((x1, y1, x2, y2, boosted_conf))

        # --- Temporal smoothing (fire persistence)
        new_fire_memory = []
        for (x1, y1, x2, y2, conf) in fire_candidates:
            matched = False
            for (fx1, fy1, fx2, fy2, old_conf) in fire_memory:
                iou = compute_iou((x1, y1, x2, y2), (fx1, fy1, fx2, fy2))
                if iou > 0.3:
                    # persist detection & increase confidence
                    conf = min(1.0, (conf + old_conf) / 2 + 0.1)
                    matched = True
                    break
            if matched or conf > 0.5:
                new_fire_memory.append((x1, y1, x2, y2, conf))

        fire_memory = new_fire_memory[-memory_limit:]

        for (x1, y1, x2, y2, conf) in fire_memory:
            detections.append((x1, y1, x2, y2, "🔥 Fire Detected", conf, (255, 100, 0)))

        # ---------------- FILTER: Remove fire overlapping faces
        filtered_detections = []
        for det in detections:
            x1, y1, x2, y2, name, conf, color = det
            if "Fire" in name:
                overlap = False
                for fd in detections:
                    if "Ben Dante" in fd[4]:
                        fx1, fy1, fx2, fy2 = fd[:4]
                        if compute_iou((x1, y1, x2, y2), (fx1, fy1, fx2, fy2)) > 0.05:
                            overlap = True
                            break
                if not overlap:
                    filtered_detections.append(det)
            else:
                filtered_detections.append(det)
        detections = filtered_detections

        # ---------------- DRAW RESULTS
        for (x1, y1, x2, y2, name, conf, color) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cvzone.putTextRect(
                frame,
                f"{name} {math.ceil(conf * 100)}%",
                [x1, y1 - 10],
                scale=1,
                thickness=2,
                colorR=color
            )

        cv2.imshow("🔥 Fire & Face Detection (YOLO - Enhanced)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_detection()
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()

# =====================================================
# 🧮 Helper: IOU Calculation
# =====================================================
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    fx1, fy1, fx2, fy2 = box2
    inter_x1 = max(x1, fx1)
    inter_y1 = max(y1, fy1)
    inter_x2 = min(x2, fx2)
    inter_y2 = min(y2, fy2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (fx2 - fx1) * (fy2 - fy1)
    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0

# =====================================================
# 🎞 GUI CONTROLS
# =====================================================
def select_video():
    file_path = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov")]
    )
    if file_path:
        start_detection(file_path)

def start_webcam():
    start_detection(0)

def start_detection(source):
    stop_detection()
    thread = threading.Thread(target=detect_fire_and_face, args=(source,))
    thread.daemon = True
    thread.start()

def stop_detection():
    global running
    running = False

# =====================================================
# 🎮 GUI
# =====================================================
root = tk.Tk()
root.title("🔥 Fire & Face Detector (Triple YOLO Gamer Mode)")
root.geometry("450x380")
root.config(bg="#0a0a0f")

header = tk.Label(
    root,
    text="🔥 FIRE & FACE DETECTOR 🔥",
    font=("Orbitron", 18, "bold"),
    fg="#00ffff",
    bg="#0a0a0f"
)
header.pack(pady=20)

def neon_button(master, text, command, color):
    return tk.Button(
        master,
        text=text,
        command=command,
        font=("Consolas", 14, "bold"),
        fg="white",
        bg=color,
        activebackground="#202040",
        activeforeground="cyan",
        relief="flat",
        bd=0,
        height=2,
        cursor="hand2"
    )

btn_video = neon_button(root, "🎞 Select Video", select_video, "#0077ff")
btn_cam = neon_button(root, "📷 Start Webcam", start_webcam, "#00cc66")
btn_stop = neon_button(root, "⏹ Stop Detection", stop_detection, "#ff0033")

btn_video.pack(fill='x', padx=50, pady=10)
btn_cam.pack(fill='x', padx=50, pady=10)
btn_stop.pack(fill='x', padx=50, pady=10)

footer = tk.Label(
    root,
    text="Press 'Q' to close the video window",
    font=("Consolas", 10),
    fg="#888",
    bg="#0a0a0f"
)
footer.pack(pady=20)

border_frame = tk.Frame(root, bg="#00ffff", height=3)
border_frame.pack(fill='x', side='bottom')

root.mainloop()
