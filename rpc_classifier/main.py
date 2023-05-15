from func import *
from rpc_classifier import *
from data import *

import math
import cv2
import configparser
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def main():
    # load config file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # cv2 webcam stream
    cv2.namedWindow("main", cv2.WINDOW_NORMAL)
    # default 640x480
    cv2.resizeWindow('main', 900, 650)
    capture = cv2.VideoCapture(0)
    with mp_hands.Hands(
            max_num_hands=config[
                'mphands'].getint('max_num_hands'),
            model_complexity=config[
                'mphands'].getint('model_complexity'),
            min_detection_confidence=config[
                'mphands'].getfloat('min_detection_confidence'),
            min_tracking_confidence=config[
                'mphands'].getfloat('min_tracking_confidence')) as hands:

        while capture.isOpened():
            # read continuous webcam input
            success, image = capture.read()
            if not success:
                print("ERROR: Empty camera frame")
                break

            # process image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # cv2 image properties
            # height=y (pixel rows), width=x (pixel columns), channels=color
            # image_height, image_width, _ = image.shape

            # draw hand annotations
            if config['debug'].getboolean('draw_hand_annotations'):
                drawHandsAnnotations(image, results)
            if config['debug'].getboolean('draw_hand_bounds'):
                drawHandsBounds(image, results)
            if config['debug'].getboolean('draw_normalized_hand'):
                drawNormalizedHands(image, results)

            # flip the image horizontally for a selfie-view display
            cv2.imshow('main', cv2.flip(image, 1))

            # exit window when pressing escape
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
