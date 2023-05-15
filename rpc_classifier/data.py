from func import *

import json
import os
import cv2
import configparser
import numpy as np
import mediapipe as mp
from tqdm import tqdm
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


config = configparser.ConfigParser()
config.read('config.ini')


def main():
    # checkData(config['data']['path'])
    checkData('.\data')
    data = dataToNormalizedCoordinates('.\data')
    saveAsJson(data, '.\data')


def checkData(data_path):
    # checks if data exists and has right file type
    print(f"\nCHECK data")
    try:
        folders = ['paper', 'rock', 'scissors']
        for folder in folders:
            folder_path = os.path.join(data_path, folder)
            files = os.listdir(folder_path)
            for file in tqdm(files):
                file_path = os.path.join(folder_path, file)
                if not checkDataType(file_path):
                    raise Exception()
                # if not checkImageDimensions(file_path):
                #    raise Exception()
            num_files = len(files)
            print(f"\nNumber of files in {folder}: {num_files}")
        print(f"\nSUCCESS: data exists")
        return True
    except:
        print(f"\nERROR: data error")
        return False


def checkImageDimensions(image_path):
    # check if the dimensions of an image are a certain value
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    if image_width != 300 or image_height != 200:
        print(
            (f"\nERROR: image has incorrect dimension {image_width}x{image_height}"))
        return False
    return True


def checkDataType(file_path):
    # check if file data type is '.png', '.jpg', '.jpeg'
    split_tup = os.path.splitext(file_path)
    file_extension = split_tup[1]
    if file_extension in ['.png', '.jpg', '.jpeg']:
        return True
    return False


def dataToNormalizedCoordinates(data_path):
    print(f"\nCONVERT data")
    data = []
    with mp_hands.Hands(
        max_num_hands=config[
            'mphands'].getint('max_num_hands'),
        model_complexity=config[
            'mphands'].getint('model_complexity'),
        min_detection_confidence=config[
            'mphands'].getfloat('min_detection_confidence'),
        min_tracking_confidence=config[
            'mphands'].getfloat('min_tracking_confidence')) as hands:

        folders = ['paper', 'rock', 'scissors']
        for folder in folders:
            gesture = []
            folder_path = os.path.join(data_path, folder)
            files = os.listdir(folder_path)

            for file in tqdm(files):
                file_path = os.path.join(folder_path, file)
                image = cv2.imread(file_path)
                results = hands.process(image)
                landmarks = getNormalizedHandLandmarks(results)
                gesture.append(landmarks)

            data.append(gesture)
            num_files = len(files)
            print(f"\nNumber of files converted for {folder}: {num_files}")

    return data


def saveAsJson(data, path):
    json_data = [[list(map(list, arr)) for arr in arr1] for arr1 in data]

    path = os.path.join(path, 'output.json')
    with open(path, 'w') as outfile:
        json.dump(json_data, outfile)


def loadFromJson(path):
    with open(path, 'r') as infile:
        json_data = json.load(infile)

    data = [[[tuple(point) for point in arr] for arr in arr1]
            for arr1 in json_data]
    return data


if __name__ == '__main__':
    main()
