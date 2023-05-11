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
    capture = cv2.VideoCapture(config['open_cv2']['video_source'])
    with mp_hands.Hands(
            max_num_hands=config['mp_hands'].getint('max_num_hands'),
            model_complexity=config['mp_hands'].getint('model_complexity'),
            min_detection_confidence=config['mp_hands'].getfloat(
                'min_detection_confidence'),
            min_tracking_confidence=config['mp_hands'].getfloat('min_tracking_confidence')) as hands:

        while capture.isOpened():
            # read continuous webcam input
            success, image = capture.read()
            if not success:
                print("ERROR: Ignoring empty camera frame.")
                continue

            # process image in different colors
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # cv2 image properties
            # height=y (pixel rows), width=x (pixel columns), channels=color
            image_height, image_width, _ = image.shape

            # draw hand annotations
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            # flip the image horizontally for a selfie-view display
            cv2.imshow('Hand Tracking', cv2.flip(image, 1))

            # exit window when pressing escape
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    capture.release()


def getLandmarkCoordinates(image, results, VISIBILITY_THRESHOLD=0.5, PRESENCE_THRESHOLD=0.5):
    # get usable coordinates for all visible landmarks
    coordinates = {}
    image_height, image_width, _ = image.shape

    try:
        for landmark_id, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
            # check if landmark is visible
            if ((landmark.HasField('visibility') and
                    landmark.visibility < VISIBILITY_THRESHOLD) or
                    (landmark.HasField('presence') and
                     landmark.presence < PRESENCE_THRESHOLD)):
                continue

            # normalize coordinates
            landmark_pixel_coords = mp_drawing._normalized_to_pixel_coordinates(
                landmark.x, landmark.y, image_width, image_height)
            if landmark_pixel_coords:
                coordinates[landmark_id] = landmark_pixel_coords
    except:
        pass

    return coordinates


if __name__ == '__main__':
    main()
