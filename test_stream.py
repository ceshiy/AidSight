import cv2

url = "http://10.88.61.160:8081/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Cannot open stream")
else:
    print("Stream opened. Reading frame...")
    ret, frame = cap.read()
    print("Read success:", ret)
    if ret:
        cv2.imwrite("test_frame.jpg", frame)
        print("Saved test_frame.jpg")
    cap.release()