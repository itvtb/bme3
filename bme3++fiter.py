import cv2
import socket
import math
from cvzone.HandTrackingModule import HandDetector
from collections import deque

cap = cv2.VideoCapture(0)
h, w = 720, 1280
cap.set(3, w)
cap.set(4, h)

detector = HandDetector(maxHands=1, detectionCon=0.8)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ("192.168.71.101", 5052)

K = 1000
depth_history = deque(maxlen=5)


landmark_history = {i: deque(maxlen=5) for i in range(21)}

while True:
    success, img = cap.read()
    if not success or img is None:
        print("No cam")
        continue

    hands, img = detector.findHands(img)
    data = []

    if hands:
        hand = hands[0]
        lmList = hand['lmList']

        x0, y0 = lmList[0][:2]
        x1, y1 = lmList[1][:2]
        d = math.dist([x0, y0], [x1, y1])
        depth_z = K / d
        depth_history.append(depth_z)
        filtered_depth = sum(depth_history) / len(depth_history)

        # fiter
        for i, lm in enumerate(lmList):
            landmark_history[i].append(lm)


            avg_x = sum(p[0] for p in landmark_history[i]) / len(landmark_history[i])
            avg_y = sum(p[1] for p in landmark_history[i]) / len(landmark_history[i])
            avg_z = sum(p[2] for p in landmark_history[i]) / len(landmark_history[i])


            data.extend([avg_x, h - avg_y, avg_z])


        data.append(filtered_depth)


        sock.sendto(str.encode(str(data)), server_address)


        # print(f"Depth_z fiter: {filtered_depth:.2f}")
        # print(f"Landmark 0 fiter: {[round(data[0],2), round(data[1],2), round(data[2],2)]}")

    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):
        break
