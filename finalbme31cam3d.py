import cv2
import socket
import math
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
h, w = 720, 1280
cap.set(3, w)
cap.set(4, h)

detector = HandDetector(maxHands=1, detectionCon=0.8)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ("192.168.71.101", 5052)

K = 1000
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    data = []

    if hands:
        hand = hands[0]
        lmList = hand['lmList']

        x0, y0 = lmList[0][:2]
        x1, y1 = lmList[1][:2]
        d = math.dist([x0, y0], [x1, y1])
        depth_z = K / d

        print(lmList)
        print(depth_z)

        for lm in lmList:
            data.extend([lm[0], h - lm[1], lm[2]])
            data.append(depth_z)
            sock.sendto(str.encode(str(data)), server_address)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
