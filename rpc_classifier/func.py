import math
import cv2
import configparser
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


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
