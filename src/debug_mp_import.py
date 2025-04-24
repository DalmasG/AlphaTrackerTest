print("⏳ Before import")
import mediapipe as mp
print("✅ Imported mediapipe:", mp.__version__)

print("⏳ Instantiating ObjectDetection…")
try:
    od = mp.solutions.object_detection.ObjectDetection(model_selection=0, min_detection_confidence=0.5)
    print("✅ ObjectDetection OK")
    od.close()
except Exception as e:
    print("❌ OD failed:", e)

print("⏳ Instantiating BlazePose…")
try:
    pose = mp.solutions.pose.Pose()
    print("✅ BlazePose OK")
    pose.close()
except Exception as e:
    print("❌ Pose failed:", e)
