import cv2
import configparser
import math
import time
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


config = configparser.ConfigParser()
config.read('config.ini')


# ----------------------------------shortcuts----------------------------------
def resultsToModelInput(image, results):
    # shortcut to get an array of arrays of normalized hand landmarks
    return flattenData(getNormalizedHandsLandmarksFlipped(image, results))


def getNormalizedHandsLandmarks(image, results):
    # shortcut to get an array of arrays of normalized hand landmarks
    return normalizeHandsLandmarksAspect(getHandsLandmarks(image, results))


def getNormalizedHandsLandmarksFlipped(image, results):
    # shortcut to get an array of arrays of normalized hand landmarks
    return normalizeHandsLandmarksAspect(getHandsLandmarksFlipped(image, results))


# ----------------------------------math----------------------------------
def toPixelCoordinates(image, point):
    # convert percentage based coordinates to absolute pixel coordinates
    image_height, image_width, _ = image.shape
    per_x, per_y = point
    pix_x, pix_y = round(per_x * image_width), round(per_y * image_height)
    return pix_x, pix_y


def toPercentageCoordinates(image, point):
    # convert absolute pixel coordinates to percentage based coordinates
    image_height, image_width, _ = image.shape
    pix_x, pix_y = point
    per_x, per_y = pix_x/image_width, pix_y/image_height
    return per_x, per_y


# ----------------------------------draw----------------------------------
def drawHandsAnnotations(image, results):
    # draw annotations for all visible hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())


def drawHandsBounds(image, results):
    # draw bounding boxes for all visible hands
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


def drawNormalizedHands(image, results):
    hands = getNormalizedHandsLandmarks(image, results)
    for i, hand in enumerate(hands):
        for point in hand:
            x, y = point
            x, y = round(x*100), round(y*100)
            center = (x, y+i*100)
            cv2.circle(image, center, 2, (255, 255, 255), -1)
            
            
def drawHandSeparator(image, results):
    # return an array of all visible hands, each of which contains an array
    # of points representing the coordinates of all visible landmarks, and flip
    # to fit into ml model
    hands = getHandsLandmarks(image, results)
    image_height, image_width, _ = image.shape
    avg_x = image_width/2
    if len(hands) > 1:
        avg_x = np.mean(np.concatenate(hands)[:, 0])*image_width
    cv2.line(image, (int(avg_x), 0), (int(avg_x), image_height), (0, 255, 0), 2)
    

def drawResultText(image, predictions):
    if len(predictions) == 2:
        gestures = ['paper', 'rock', 'scissors']
        pred_text = []
        for i in predictions:
            pred_text.append(gestures[i])
    
        image_height, image_width, _ = image.shape
        cv2.putText(img=image, text=pred_text[0], org=(int(image_width/2-5), int(image_height-5)),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=-1.5,
                    color=(0, 255, 0), thickness=2, bottomLeftOrigin=True)
        cv2.putText(img=image, text=pred_text[1], org=(int(image_width-5), int(image_height-5)),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=-1.5,
                    color=(0, 255, 0), thickness=2, bottomLeftOrigin=True)
    else:
        return
    
    
def drawPlayerNames(image):
    _, image_width, _ = image.shape
    cv2.putText(img=image, text="Player 1", org=(int(image_width/4+50), int(15)),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=-1.5,
                    color=(255, 255, 255), thickness=2, bottomLeftOrigin=True)
    cv2.putText(img=image, text="Player 2", org=(int(image_width/4*3+50), int(15)),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=-1.5,
                    color=(255, 255, 255), thickness=2, bottomLeftOrigin=True)


# ----------------------------------coords----------------------------------
def getHandsLandmarks(image, results):
    # return an array of all visible hands, each of which contains an array
    # of points representing the coordinates of all visible landmarks
    hands = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            if hand_landmarks:
                for i in range(21):
                    landmark = hand_landmarks.landmark[i]
                    x = (landmark.x if landmark.x > 0 else 0)
                    y = (landmark.y if landmark.y > 0 else 0)
                    landmark_coords = (x, y)
                    landmarks.append(landmark_coords)
            hands.append(landmarks)
    return hands


def getHandsLandmarksFlipped(image, results):
    # return an array of all visible hands, each of which contains an array
    # of points representing the coordinates of all visible landmarks, and flip
    # to fit into ml model
    hands = getHandsLandmarks(image, results)
    hands_flipped = []
    avg_x = 0.5
    if len(hands) > 1:
        avg_x = np.mean(np.concatenate(hands)[:, 0])
        
    for hand in hands:
        hand_flipped = hand
        avg_x_hand = np.mean(np.array(hand)[:, 0])
        if avg_x_hand < avg_x:
            hand_flipped = flipHand(hand)
        hands_flipped.append(hand_flipped)
    #print("f", [np.mean(np.array(arr)[:, 0]) for arr in hands_flipped])
    hands_sorted = sorted(hands_flipped, key=lambda arr: np.mean(np.array(arr)[:, 0]))
    #print("s", [np.mean(np.array(arr)[:, 0]) for arr in hands_sorted])
    return hands_sorted
        
                     
def flipHand(hand):
    # flip hand coords
    max_x = max(point[0] for point in hand)
    hand_flipped = [(2*max_x - point[0], point[1]) for point in hand]
    return hand_flipped   


# ----------------------------------normalize----------------------------------
def normalizeHandsLandmarksAspect(hands):
    # normalizes the coordinates for an array of all visible hands, so that
    # each hand is represented in the same size (0 to 1) regardless of on-screen size
    norm_hands = []
    if hands:
        for hand in hands:
            norm_hand = normalizeHandLandmarksAspect(hand)
            norm_hands.append(norm_hand)
    return norm_hands


def normalizeHandLandmarks(hand):
    # normalizes the coordinates for a single hand array of points to between 0 to 1
    try:
        min_h0, max_h0 = np.min(hand[:, 0]), np.max(hand[:, 0])
        min_h1, max_h1 = np.min(hand[:, 1]), np.max(hand[:, 1])
        delta0, delta1 = max_h0-min_h0, max_h1-min_h1
        min_h, max_h = min_h0, max_h0 if delta0 > delta1 else max_h1, max_h1

        norm_h = (hand - min_h) / (max_h - min_h)
        norm_hand = []
        for coordinates in norm_h:
            point = coordinates[0], coordinates[1]
            norm_hand.append(point)
        return norm_hand
    except:
        pass


def normalizeHandLandmarksAspect(hand):
    # normalizes the coordinates for a single hand array of points to between 0 to 1
    try:
        arr = np.array(hand)

        x_range = np.max(arr[:, 0]) - np.min(arr[:, 0])
        y_range = np.max(arr[:, 1]) - np.min(arr[:, 1])
        max_range = max(x_range, y_range)

        min_val = np.min(arr, axis=0)
        norm_h = (arr - min_val) / max_range

        norm_hand = []
        for coordinates in norm_h:
            point = coordinates[0], coordinates[1]
            norm_hand.append(point)
        return norm_hand
    except:
        pass


# ----------------------------------flatten----------------------------------
def flattenData(data):
    data_flat = []
    if data: 
        for hand in data:
            hand_flat = []
            for point in hand:
                x, y = point
                hand_flat.append(x)
                hand_flat.append(y)
            data_flat.append(hand_flat)
    return data_flat


# ----------------------------------:)----------------------------------
def avgCoordinates(array, index):
    arr = array[index]
    total_points = len(arr) // 2
    sum_x = sum(arr[i] for i in range(0, len(arr), 2))
    sum_y = sum(arr[i] for i in range(1, len(arr), 2))
    avgx = sum_x / total_points
    avgy = sum_y / total_points
    return (avgx, avgy)


def maxDistance(points):
    max_distance = 0
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            distance = math.dist(points[i], points[j])
            if distance > max_distance:
                max_distance = distance
    return max_distance


def evaluateResults(image, predictions):
    # p1 lower x value, p2 higherx value
    # [0='paper', 1='rock', 2='scissors']
    image_height, image_width, _ = image.shape
    text = ''
    p1, p2 = predictions[0], predictions[1]
    if (p1 == 0 and p2 == 1 
            or p1 == 1 and p2 == 2 
            or p1 == 2 and p2 == 0):
        text = 'Player 1 wins!'
    elif (p2 == 0 and p1 == 1 
            or p2 == 1 and p1 == 2 
            or p2 == 2 and p1 == 0):
        text = 'Player 2 wins!'
    else:
        text = 'Draw!'     
    return text


def putTextCenter(image, text, font_scale, color, thickness):
    font=cv2.FONT_HERSHEY_PLAIN
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size

    # Calculate the center position
    image_height, image_width = image.shape[:2]
    center_x = (image_width - text_width) // 2
    center_y = (image_height + text_height) // 2

    # Draw the text at the center position
    cv2.putText(image, text, (center_x, center_y), 
                font, font_scale, color, thickness, 
                bottomLeftOrigin=True)