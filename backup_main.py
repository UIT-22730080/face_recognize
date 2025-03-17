import cv2
import face_recognition
import os
import numpy as np

# Thư mục lưu ảnh khuôn mặt
FACES_DIR = 'faces'
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

def capture_and_save_face(user_id, num_photos=5):
    """Chụp và lưu nhiều ảnh khuôn mặt của người dùng."""
    user_dir = os.path.join(FACES_DIR, user_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    cap = cv2.VideoCapture(1)  # Thay đổi ID camera nếu cần
    
    if not cap.isOpened():
        print("Không thể mở camera!")
        return
    
    print("Nhấn 's' để chụp ảnh, 'q' để thoát.")

    photo_count = 0

    while photo_count < num_photos:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc dữ liệu từ camera!")
            break

        # Chuyển ảnh sang RGB để nhận diện khuôn mặt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        # for (top, right, bottom, left) in face_locations:
            # Vẽ khung chữ nhật màu xanh bao quanh khuôn mặt
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(frame, f"Press 'S' to capture ({photo_count+1}/{num_photos}), 'Q' to quit", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow('Capture Face', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if face_locations:
                face_image_path = os.path.join(user_dir, f'{user_id}_{photo_count+1}.jpg')
                cv2.imwrite(face_image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"✅ Đã lưu ảnh khuôn mặt vào {face_image_path}")
                photo_count += 1
            else:
                print("⚠ Không phát hiện khuôn mặt, hãy chụp lại.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def load_known_faces():
    """Tải và mã hóa các khuôn mặt đã lưu."""
    known_face_encodings = []
    known_face_names = []

    for user_id in os.listdir(FACES_DIR):
        user_dir = os.path.join(FACES_DIR, user_id)
        if os.path.isdir(user_dir):
            for image_name in os.listdir(user_dir):
                image_path = os.path.join(user_dir, image_name)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(user_id)
                else:
                    print(f"⚠ Không tìm thấy khuôn mặt trong ảnh {image_name}.")

    return known_face_encodings, known_face_names

def recognize_faces():
    video_capture = cv2.VideoCapture(1)  # Thay đổi ID nếu cần
    known_face_encodings, known_face_names = load_known_faces()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Không thể lấy hình ảnh từ camera.")
            continue

        # Thay đổi kích thước khung hình
        frame = cv2.resize(frame, (640, 480))
        rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)

        if face_locations:  # Kiểm tra có khuôn mặt không
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Không xác định"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    print(f"Phát hiện khuôn mặt ... [ {name} ]")
                    # video_capture.release()
                    # cv2.destroyAllWindows()
                    # return

                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # else:
        #     print("⚠ Không phát hiện khuôn mặt, hãy thử lại.")

        cv2.imshow('Nhận diện khuôn mặt', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # user_id = input("Nhập ID người dùng: ")
    # capture_and_save_face(user_id)
    recognize_faces()