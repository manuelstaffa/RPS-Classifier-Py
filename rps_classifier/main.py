from func import *
from data import *
from rps_classifier import *

import time
import cv2
import configparser
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


config = configparser.ConfigParser()
config.read('config.ini')


def main():
    # load model
    path = config['data']['path']
    svm = loadSVM(path, 'svm_classifier.pkl')
    
    start_time = time.time()
    points_h1 = []
    points_h2 = []
    stationary = False
    predicted = False
    result_text = ''

    # cv2 webcam stream
    cv2.namedWindow("main", cv2.WINDOW_NORMAL)
    # default 640x480
    cv2.resizeWindow('main', 1200, 900)
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
            drawPlayerNames(image, -1.5, (255, 255, 255), 2)

            # pass results to ml model
            # [0='paper', 1='rock', 2='scissors']
            # remember final image is flipped
            current_time = time.time()
            detection_frequency = config[
                'evaluate'].getfloat('detection_frequency')
            detection_time = config[
                'evaluate'].getfloat('detection_time')
            movement_accuracy = config[
                'evaluate'].getfloat('movement_accuracy')
            win_time = config[
                'evaluate'].getfloat('win_time')
            
            X = resultsToModelInput(image, results)
            predictions = []
            if X:
                predictions = svm.predict(X)
            
                if current_time - start_time > detection_frequency and len(X) == 2:
                    start_time = current_time
                    
                    amount = detection_time/detection_frequency
                    points_h1.append(avgCoordinates(X, 0)) 
                    if len(points_h1) > amount:
                        points_h1.pop(0)
                    points_h2.append(avgCoordinates(X, 1)) 
                    if len(points_h2) > amount:
                        points_h2.pop(0)
                        
                    if (len(points_h1) >= amount 
                            and len(points_h2) >= amount 
                            and maxDistance(points_h1) < movement_accuracy 
                            and maxDistance(points_h2) < movement_accuracy):
                        stationary = True
                
                if stationary and not predicted:        
                    stationary = False
                    predicted = True
                    result_text = evaluateResults(image, predictions)
                    eval_time = current_time
                    #print(predictions, result_text)
            
            result_text_time = ""      
            if predicted:
                result_text_time = result_text + f" [{win_time-round(current_time - eval_time)}s]"
                if current_time - eval_time > win_time:
                    stationary = False
                    predicted = False
                    points_h1.clear()
                    points_h2.clear() 
                    result_text = ''
                    
            drawTextCenter(image, result_text_time, -2, (255, 255, 255), 2)          
            
            # draw hand annotations
            if config['debug'].getboolean('draw_hand_annotations'):
                drawHandsAnnotations(image, results)
            if config['debug'].getboolean('draw_hand_bounds'):
                drawHandsBounds(image, results)
            if config['debug'].getboolean('draw_normalized_hand'):
                drawNormalizedHands(image, results)
            if config['debug'].getboolean('draw_separator'):
                drawHandSeparator(image, results)
            if config['debug'].getboolean('draw_predictions'):
                drawResultText(image, predictions)  
                
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
