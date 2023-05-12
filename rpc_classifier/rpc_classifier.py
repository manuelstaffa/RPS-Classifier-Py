from func import *
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
    config.sections()
    config.read('config.ini')

    # cv2 webcam stream
    cv2.namedWindow("main", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('main', 900, 900)
    capture = cv2.VideoCapture(config['open_cv2'].getint('video_source'))
    with mp_hands.Hands(
            max_num_hands=config[
                'mp_hands'].getint('max_num_hands'),
            model_complexity=config[
                'mp_hands'].getint('model_complexity'),
            min_detection_confidence=config[
                'mp_hands'].getfloat('min_detection_confidence'),
            min_tracking_confidence=config[
                'mp_hands'].getfloat('min_tracking_confidence')) as hands:

        while capture.isOpened():
            # read continuous webcam input
            success, image = capture.read()
            if not success:
                print("ERROR: Ignoring empty camera frame.")
                continue

            # process image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # cv2 image properties
            # height=y (pixel rows), width=x (pixel columns), channels=color
            image_height, image_width, _ = image.shape

            # draw hand annotations
            drawHandAnnotations(image, results)

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
