import math
import cv2
import configparser
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


config = configparser.ConfigParser()
config.read('config.ini')


def toPixelCoordinates(image, point):
    image_height, image_width, _ = image.shape
    x, y = point
    norm_x, norm_y = round(x * image_width), round(y * image_height)
    return norm_x, norm_y


def drawHandAnnotations(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())


def drawHandBounds(image, results):
    if results.multi_hand_landmarks:
        for hand, hand_landmarks in enumerate(results.multi_hand_landmarks):
            x, y = [], []
            if hand_landmarks:
                for i in range(21):
                    landmark = hand_landmarks.landmark[i]
                    x.append(landmark.x if landmark.x > 0 else 0)
                    y.append(landmark.y if landmark.y > 0 else 0)
            p_min = toPixelCoordinates(image, (min(x), min(y)))
            p_max = toPixelCoordinates(image, (max(x), max(y)))
            cv2.rectangle(image, p_min, p_max, (0, 255, 0), 2)
            # cv2.line(image, p_min, (0, 0), (0, 255, 0), 2)
            # cv2.line(image, p_max, (0, 0), (0, 255, 0), 2)
            cv2.putText(img=image, text="Hand "+str(hand), org=(p_max[0], p_min[1]),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=-1,
                        color=(0, 255, 0), thickness=2, bottomLeftOrigin=True)
