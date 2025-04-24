import cv2

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ ERROR: Cannot open camera")
        return

    print("✅ Camera opened. Press ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Frame not received")
            break

        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Test ended.")

if __name__ == "__main__":
    main()
