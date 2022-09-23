import cv2
from pypylon import pylon
from pypylon_opencv_viewer import BaslerOpenCVViewer
import re
import numpy as np
from matplotlib import pyplot as plt
import sys
import mediapipe as mp
import json
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.8,modelComplexity=0,trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)


        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self,image, handNo = 0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            t = str(self.results.multi_handedness)
            #print(t)
            if ("Left" in t) & ("Right" not in t):
                for id, lm in enumerate(Hand.landmark):
                    h,w,c = image.shape
                    cx,cy = int(lm.x*w), int(lm.y*h)
                    lmlist.append([id,cx,cy])
                    #print(f'{id},{cx},{cy}')
                    cv2.circle(image,(cx,cy), 10 , (0,255,0), cv2.FILLED)
                    if id == 0:
                        font = cv2.FONT_HERSHEY_COMPLEX
                        fontScale = 1
                        thickness = 3
                        org5 = (cx, cy)
                        t = str(self.results.multi_handedness).split(" ")
                        cv2.putText(image, str(t[10].split("}\n",1)[0]), org5, font, fontScale, (0,255,0), thickness, 3)

        return lmlist

# For webcam input:
cap = cv2.VideoCapture(0)
tracker = handTracker()
# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0
while True:
    success, image = cap.read()
    image
    image = tracker.handsFinder(image)
    lmList = tracker.positionFinder(image)

    # Calculating the FPS
    currentTime = time.time()

    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
    cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Video", image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
