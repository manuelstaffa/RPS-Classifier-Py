import math
import cv2
import configparser
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


config = configparser.ConfigParser()
config.sections()
config.read('config.ini')


def getLandmarkCoordinates(image, results):
    # get usable coordinates for all visible landmarks
    coordinates = {}
    image_height, image_width, _ = image.shape

    try:
        for landmark_id, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
            # check if landmark is visible
            if ((landmark.HasField('visibility') and
                landmark.visibility < config[
                    'mp.landmarks'].getfloat('visibility_threshold')) or
                (landmark.HasField('presence') and
                 landmark.presence < config[
                    'mp.landmarks'].getfloat('presence_threshold'))):
                continue

            # normalize coordinates
            landmark_pixel_coords = mp_drawing._normalized_to_pixel_coordinates(
                landmark.x, landmark.y, image_width, image_height)
            if landmark_pixel_coords:
                coordinates[landmark_id] = landmark_pixel_coords
    except:
        pass

    return coordinates


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
    return
