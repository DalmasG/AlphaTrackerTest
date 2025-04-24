import cv2

def main():
    # Try opening camera (no CAP_DSHOW)
    cap = cv2.VideoCapture(0)
    print("Camera open status:", cap.isOpened())

    # Try reading 10 frames
    for i in range(10):
        ret, frame = cap.read()
        print(f"Frame {i}: ret={ret}", end="")
        if ret:
            print(f", shape={frame.shape}")
        else:
            print()
    cap.release()

if __name__ == "__main__":
    main()
