import cv2
import socket
from cvzone.HandTrackingModule import HandDetector
cap = cv2.VideoCapture(1)
h,w = 720,1280
cap.set(3, w)
cap.set(4, h)
detector = HandDetector(maxHands= 1, detectionCon= 0.8)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ("10.136.213.143", 5052)
while True:
    success,  img  = cap.read()
    hands, img = detector.findHands(img)
    data = []
    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        print(lmList)
        for lm in lmList:
            data.extend([lm[0], h - lm[1], lm[2]])
            sock.sendto(str.encode(str(data)), server_address)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
