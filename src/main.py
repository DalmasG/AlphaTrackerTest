import cv2
import time
import logging
from collections import deque
import mediapipe as mp

# ─── Settings ───────────────────────────────────────────────────────────────────
CAM_INDEX        = 0
WIDTH, HEIGHT    = 640, 360
TARGET_FPS       = 15
VISIBILITY_THRESH= 0.4
PADDING          = 20    # pixels of padding around detected person boxes

# BlazePose model settings
MODEL_COMPLEXITY = 0
MIN_CONF         = 0.5   # threshold for both SSD and BlazePose

# Paths to your downloaded SSD files
PROTO_TXT = "MobileNetSSD_deploy.prototxt"
MODEL     = "MobileNetSSD_deploy.caffemodel"

# SSD class IDs: background=0, person=15
PERSON_CLASS_ID = 15

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename='tracking_log.txt',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

# ─── MediaPipe utilities ─────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def main():
    print("Starting SSD+BlazePose demo with padded boxes…")

    # 1) Load SSD model
    net = cv2.dnn.readNetFromCaffe(PROTO_TXT, MODEL)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 2) Open camera
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        print("❌ ERROR: Cannot open camera")
        return

    # Create a resizable window and set fullscreen
    cv2.namedWindow("Grappling Tracker", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Grappling Tracker",
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    # 3) Init BlazePose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        enable_segmentation=False,
        min_detection_confidence=MIN_CONF,
        min_tracking_confidence=MIN_CONF
    )

    fps_buf = deque(maxlen=30)
    total_frames = 0
    low_drop_frames = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to read frame")
            break
        total_frames += 1
        h, w = frame.shape[:2]

        # ── SSD person detection with padding ─────────────────────────────────
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            score = float(detections[0,0,i,2])
            cls   = int(detections[0,0,i,1])
            if cls == PERSON_CLASS_ID and score > MIN_CONF:
                # raw box coordinates
                rx1 = int(detections[0,0,i,3] * w)
                ry1 = int(detections[0,0,i,4] * h)
                rx2 = int(detections[0,0,i,5] * w)
                ry2 = int(detections[0,0,i,6] * h)
                # padded box
                x1 = max(0, rx1 - PADDING)
                y1 = max(0, ry1 - PADDING)
                x2 = min(w, rx2 + PADDING)
                y2 = min(h, ry2 + PADDING)
                boxes.append((x1, y1, x2, y2))
                # draw padded box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                cv2.putText(frame, f"person {score:.2f}",
                            (x1, max(0, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # ── BlazePose per padded ROI ─────────────────────────────────────────────
        low_vis = 0
        for (x1, y1, x2, y2) in boxes:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            res = pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame[y1:y2, x1:x2],
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                for lm in res.pose_landmarks.landmark:
                    if lm.visibility < VISIBILITY_THRESH:
                        low_vis += 1
        if low_vis > 3:
            low_drop_frames += 1

        # ── FPS & display ─────────────────────────────────────────────────────
        now = time.time()
        inst_fps = 1 / (now - prev_time + 1e-6)
        prev_time = now
        fps_buf.append(inst_fps)
        avg_fps = sum(fps_buf) / len(fps_buf)

        cv2.putText(frame, f"FPS: {inst_fps:.1f} ({avg_fps:.1f} avg)",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, f"Drops: {low_vis}",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        logging.info(f"frame={total_frames} drops={low_vis} fps={inst_fps:.2f}")

        cv2.imshow("Grappling Tracker", frame)
        if cv2.waitKey(int(1000/TARGET_FPS)) & 0xFF == 27:
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    pct = low_drop_frames / total_frames * 100 if total_frames else 0
    print(f"\nSummary: {total_frames} frames, {low_drop_frames} high-drop frames ({pct:.1f}% )")

if __name__ == "__main__":
    main()