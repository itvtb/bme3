import cv2
import numpy as np
import mediapipe as mp

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Mở hai camera
video_capture_0 = cv2.VideoCapture(1)
video_capture_1 = cv2.VideoCapture(2)

# Thông số camera (giả định từ hiệu chuẩn)
camera_matrix_0 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
camera_matrix_1 = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs_0 = np.zeros(5)
dist_coeffs_1 = np.zeros(5)

# Vị trí tương đối của Cam1 so với Cam0
R = cv2.Rodrigues(np.array([0, np.deg2rad(30), 0]))[0]  # Ma trận xoay
T = np.array([[0.3], [0], [0]], dtype=np.float32)  # Dịch chuyển 30 cm
R_inv = np.linalg.inv(R)  # Ma trận xoay nghịch đảo

# Hàm tam giác hóa
def triangulate_points(pt0, pt1, P0, P1):
    points_4d = cv2.triangulatePoints(P0, P1, pt0, pt1)
    points_3d = points_4d[:3] / points_4d[3]
    return points_3d.T

# Hàm chuyển tọa độ từ hệ Cam0 sang hệ Cam1
def transform_to_cam1(points_3d, R_inv, T):
    points_cam1 = []
    for point in points_3d:
        # Dịch chuyển: P - T
        point_translated = point.reshape(3, 1) - T
        # Xoay: R^-1 * (P - T)
        point_cam1 = R_inv @ point_translated
        points_cam1.append(point_cam1.flatten())
    return np.array(points_cam1)

# Thiết lập Hand Detector
with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    # Ma trận chiếu
    P0 = camera_matrix_0 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = camera_matrix_1 @ np.hstack((R, T))

    while True:
        ret0, frame0 = video_capture_0.read()
        ret1, frame1 = video_capture_1.read()

        points_2d_0 = []
        points_2d_1 = []
        landmark_indices = []

        if ret0:
            image0 = cv2.flip(frame0, 1)
            image0_rgb = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
            results0 = hands.process(image0_rgb)

            if results0.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results0.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image0, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for lm_idx, lm in enumerate(hand_landmarks.landmark):
                        points_2d_0.append([lm.x * image0.shape[1], lm.y * image0.shape[0]])
                        landmark_indices.append(f"Hand_{hand_idx}_Landmark_{lm_idx}")

            cv2.imshow('Cam 0 - Hand Tracking', image0)

        if ret1:
            image1 = cv2.flip(frame1, 1)
            image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            results1 = hands.process(image1_rgb)

            if results1.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results1.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for lm_idx, lm in enumerate(hand_landmarks.landmark):
                        points_2d_1.append([lm.x * image1.shape[1], lm.y * image1.shape[0]])

            cv2.imshow('Cam 1 - Hand Tracking', image1)

        # Tam giác hóa và chuyển tọa độ
        if points_2d_0 and points_2d_1 and len(points_2d_0) == len(points_2d_1):
            points_2d_0 = np.array(points_2d_0, dtype=np.float32).T
            points_2d_1 = np.array(points_2d_1, dtype=np.float32).T
            points_3d = triangulate_points(points_2d_0, points_2d_1, P0, P1)

            # Chuyển tọa độ sang hệ Cam1
            points_3d_cam1 = transform_to_cam1(points_3d, R_inv, T)

            # In tọa độ 3D trong hệ Cam1
            print("\n3D Coordinates (Origin at Cam1, in meters):")
            for i, (point, label) in enumerate(zip(points_3d_cam1, landmark_indices)):
                print(f"{label}: X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f}")

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng tài nguyên
video_capture_0.release()
video_capture_1.release()
cv2.destroyAllWindows()