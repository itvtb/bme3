import cv2
import mediapipe as mp
import mysql.connector
from datetime import datetime

# Kết nối MySQL (SỬA TÊN DATABASE)
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # Để trống nếu dùng XAMPP mặc định
    database="Phuchoichucnang_01"  # <-- SỬA TÊN DATABASE
)
cursor = db.cursor()

# Khởi tạo MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Không thể đọc khung hình.")
            continue

        # Xử lý hình ảnh
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Vẽ điểm mốc và gửi dữ liệu lên MySQL
        if results.pose_landmarks:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Lấy thời gian hiện tại

            for idx in range(11, 23):  # Lặp từ điểm 11 đến 22
                landmark = results.pose_landmarks.landmark[idx]
                if landmark.visibility > 0.5:
                    # Lấy giá trị x, y, visibility
                    x = landmark.x
                    y = landmark.y
                    visibility = landmark.visibility

                    # Chèn dữ liệu vào MySQL (SỬA TÊN TABLE)
                    sql = """
                        INSERT INTO camera1  # <-- SỬA TÊN TABLE
                        (timestamp, landmark_id, x, y, visibility)
                        VALUES (%s, %s, %s, %s, %s)
                    """
                    values = (current_time, idx, x, y, visibility)
                    cursor.execute(sql, values)
                    db.commit()  # Lưu dữ liệu

            # Vẽ điểm mốc lên hình ảnh
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        # Hiển thị hình ảnh
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Giải phóng tài nguyên
cursor.close()
db.close()
cap.release()
cv2.destroyAllWindows()