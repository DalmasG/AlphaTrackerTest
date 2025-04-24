print("=== detect_person.py starting ===")

import cv2
import mediapipe as mp

# 1) Setup camera on DirectShow at 640×360
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

if not cap.isOpened():
    print("❌ ERROR: Cannot open camera")
    exit()
print("✅ Camera opened (640×360). Press ESC to quit.")

# 2) Initialize MediaPipe person detector
mp_od = mp.solutions.object_detection.ObjectDetection(
    model_selection=0,
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# 3) Loop & draw boxes
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Failed to read frame")
        break

    # Detect
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_od.process(rgb)

    h, w = frame.shape[:2]
    if results.detections:
        for det in results.detections:
            cat = det.categories[0]
            if cat.category_name == "person":
                r = det.location_data.relative_bounding_box
                x1 = int(r.xmin * w)
                y1 = int(r.ymin * h)
                x2 = int((r.xmin + r.width) * w)
                y2 = int((r.ymin + r.height) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame,
                            f"person {cat.score:.2f}",
                            (x1, max(0,y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2)

    cv2.imshow("Person Detection Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 detect_person.py exiting.")
