import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import face_recognition
import os
import main
import dlib
import numpy as np
import threading
from queue import Queue
import time

# Tải mô hình phát hiện điểm đặc trưng trên khuôn mặt
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    raise FileNotFoundError(f"Tệp {predictor_path} không được tìm thấy. Vui lòng tải xuống và đặt vào thư mục dự án.")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

class CustomDialog(simpledialog.Dialog):
    def body(self, master):
        self.geometry("300x100")
        tk.Label(master, text="Nhập ID người dùng:").grid(row=0)
        self.user_id_entry = tk.Entry(master)
        self.user_id_entry.grid(row=0, column=1)
        return self.user_id_entry

    def apply(self):
        self.result = self.user_id_entry.get()

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống nhận diện khuôn mặt")
        self.root.geometry("800x600")

        # Tạo frame cho các nút chức năng
        self.button_frame = tk.Frame(root, width=200, bg="#f0f0f0")
        self.button_frame.pack(side="left", fill="y")

        self.label = tk.Label(self.button_frame, text="Chức năng", font=("Helvetica", 16), bg="#f0f0f0")
        self.label.pack(pady=20)

        self.capture_button = tk.Button(self.button_frame, text="Thêm khuôn mặt", command=self.capture_face, bg="#4CAF50", fg="white", font=("Arial", 12))
        self.capture_button.pack(pady=10, fill="x")

        self.recognize_button = tk.Button(self.button_frame, text="Nhận diện khuôn mặt", command=self.recognize_face, bg="#2196F3", fg="white", font=("Arial", 12))
        self.recognize_button.pack(pady=10, fill="x")

        self.manage_button = tk.Button(self.button_frame, text="Quản lí thông tin", command=self.manage_faces, bg="#FF9800", fg="white", font=("Arial", 12))
        self.manage_button.pack(pady=10, fill="x")

        self.exit_button = tk.Button(self.button_frame, text="Thoát", command=root.quit, bg="#f44336", fg="white", font=("Arial", 12))
        self.exit_button.pack(pady=10, fill="x")

        # Tạo frame cho màn hình hiển thị
        self.display_frame = tk.Frame(root, bg="#ffffff")
        self.display_frame.pack(side="right", fill="both", expand=True)

        self.cap = None
        self.current_mode = None
        self.processing = False
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.blink_count = {}  # Thêm dictionary để đếm số lần chớp mắt cho mỗi người

    def capture_face(self):
        dialog = CustomDialog(self.root)
        user_id = dialog.result
        if user_id:
            main.capture_and_save_face(user_id)
            messagebox.showinfo("Info", f"Đã chụp và lưu ảnh khuôn mặt cho ID: {user_id}")

    def recognize_face(self):
        self.current_mode = "recognize"
        self.blink_count = {}  # Reset số lần chớp mắt khi bắt đầu nhận diện mới
        self.open_camera()
        if self.cap is None:
            return
        
        # Bắt đầu luồng xử lý camera
        self.processing = True
        camera_thread = threading.Thread(target=self.camera_stream)
        process_thread = threading.Thread(target=self.process_frames)
        camera_thread.daemon = True
        process_thread.daemon = True
        camera_thread.start()
        process_thread.start()
        
        # Bắt đầu hiển thị kết quả
        self.show_recognition_window()

    def camera_stream(self):
        while self.processing and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            time.sleep(0.01)  # Giảm tải CPU

    def process_frames(self):
        known_face_encodings, known_face_names = main.load_known_faces()
        while self.processing:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                frame = cv2.resize(frame, (640, 480))
                results = main.process_frame(frame, known_face_encodings, known_face_names)
                
                # Hiển thị số lần chớp mắt của tất cả người dùng ở góc trái
                y_offset = 30  # Vị trí bắt đầu
                for name, count in self.blink_count.items():
                    cv2.putText(frame, f"{name}: Đã chớp mắt {count} lần", 
                              (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 0), 2)
                    y_offset += 30  # Khoảng cách giữa các dòng
                
                # Vẽ kết quả nhận diện lên frame
                for result in results:
                    top, right, bottom, left = result['location']
                    name = result['name']
                    blink_detected = result['blink_detected']

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Khởi tạo số lần chớp mắt nếu chưa có
                    if name not in self.blink_count:
                        self.blink_count[name] = 0
                    
                    # Tăng số lần chớp mắt nếu phát hiện
                    if blink_detected:
                        self.blink_count[name] += 1
                    
                    # Hiển thị tên người dùng
                    cv2.putText(frame, name, (left, top - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if not self.result_queue.full():
                    self.result_queue.put(frame)
            time.sleep(0.01)

    def show_recognition_window(self):
        if not self.processing:
            return

        if not self.result_queue.empty():
            frame = self.result_queue.get()
            cv2.imshow('Face Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.close_camera()
            return

        self.root.after(10, self.show_recognition_window)

    def manage_faces(self):
        for widget in self.display_frame.winfo_children():
            widget.destroy()

        manage_label = tk.Label(self.display_frame, text="Quản lí thông tin khuôn mặt", font=("Helvetica", 16))
        manage_label.pack(pady=20)

        columns = ("ID", "Số lượng ảnh")
        tree = ttk.Treeview(self.display_frame, columns=columns, show="headings")
        tree.heading("ID", text="ID")
        tree.heading("Số lượng ảnh", text="Số lượng ảnh")

        for user_id in os.listdir(main.FACES_DIR):
            user_dir = os.path.join(main.FACES_DIR, user_id)
            if os.path.isdir(user_dir):
                num_photos = len(os.listdir(user_dir))
                tree.insert("", tk.END, values=(user_id, num_photos))

        tree.pack(pady=10, fill="both", expand=True)

        delete_button = tk.Button(self.display_frame, text="Xóa khuôn mặt", command=lambda: self.delete_face(tree))
        delete_button.pack(pady=10)

    def delete_face(self, tree):
        selected_item = tree.selection()
        if selected_item:
            user_id = tree.item(selected_item, "values")[0]
            if main.delete_face(user_id):
                tree.delete(selected_item)
                messagebox.showinfo("Info", f"Đã xóa khuôn mặt của người dùng: {user_id}")

    def open_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(1)  # Đổi từ 1 thành 0
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Không thể mở camera!")
                self.cap = None

    def close_camera(self):
        self.processing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.current_mode = None
            cv2.destroyAllWindows()
            # Xóa hết queue
            while not self.frame_queue.empty():
                self.frame_queue.get()
            while not self.result_queue.empty():
                self.result_queue.get()

    def __del__(self):
        self.close_camera()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()