import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
import cv2
import face_recognition
import os
import main
import tkinter.ttk as ttk

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

    def capture_face(self):
        dialog = CustomDialog(self.root)
        user_id = dialog.result
        if user_id:
            main.capture_and_save_face(user_id)
            messagebox.showinfo("Info", f"Đã chụp và lưu ảnh khuôn mặt cho ID: {user_id}")

    def recognize_face(self):
        main.recognize_faces()

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
            user_dir = os.path.join(main.FACES_DIR, user_id)
            if os.path.exists(user_dir):
                for file in os.listdir(user_dir):
                    os.remove(os.path.join(user_dir, file))
                os.rmdir(user_dir)
                tree.delete(selected_item)
                messagebox.showinfo("Info", f"Đã xóa khuôn mặt của người dùng: {user_id}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()