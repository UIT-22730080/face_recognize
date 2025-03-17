import cv2
index = 0
while True:
    cap = cv2.VideoCapture(index)
    if cap.read()[0]:
        print(f"Camera found at index {index}")
    cap.release()
    index += 1
    if index > 10:  # Giới hạn tìm kiếm
        break
