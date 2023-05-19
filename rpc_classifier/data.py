from func import *

import json
import os
import cv2
import configparser
import mediapipe as mp
from tqdm import tqdm
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


config = configparser.ConfigParser()
config.read('config.ini')


def main():
    path = config['data']['path']
    checkData(path)
    data = dataToNormalizedCoordinates(path)
    saveAsJson(data, path)
    # data = loadFromJson(path)


def checkData(data_path):
    # checks if data exists and has right file type
    try:
        folders = ['paper', 'rock', 'scissors']
        for folder in folders:
            folder_path = os.path.join(data_path, folder)
            files = os.listdir(folder_path)

            for file in tqdm(files):
                file_path = os.path.join(folder_path, file)
                if not checkDataType(file_path):
                    raise Exception()

            num_files = len(files)
            print(f"\nNumber of files in {folder}: {num_files}")
        return True
    except:
        print(f"\nERROR: data does not exists in this location")
        return False


def checkDataType(file_path):
    # check if file data type is '.png', '.jpg', '.jpeg'
    split_tup = os.path.splitext(file_path)
    file_extension = split_tup[1]
    if file_extension in ['.png', '.jpg', '.jpeg']:
        return True
    print(f"\nERROR: file with wrong type found")
    return False


def dataToNormalizedCoordinates(data_path):
    # converts data while removing empties
    data = []
    with mp_hands.Hands(
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1) as hands:

        folders = ['paper', 'rock', 'scissors']
        for folder in folders:
            folder_path = os.path.join(data_path, folder)
            files = os.listdir(folder_path)

            gesture = []
            for file in tqdm(files):
                file_path = os.path.join(folder_path, file)
                image = cv2.imread(file_path)

                results = hands.process(image)
                landmarks = getNormalizedHandsLandmarks(results)
                if landmarks:
                    gesture.append(landmarks[0])

            data.append(gesture)
            num_files = len(files)
            num_conv = len(gesture)
            print(
                f"\nNumber of files converted for {folder}: {num_conv} of {num_files}")
    return data


def saveAsJson(data, path):
    json_data = [[list(map(list, arr)) for arr in arr1] for arr1 in data]

    path = os.path.join(path, 'output.json')
    with open(path, 'w') as outfile:
        json.dump(json_data, outfile)


def loadFromJson(path):
    path = os.path.join(path, 'output.json')
    with open(path, 'r') as infile:
        json_data = json.load(infile)

    data = [[[tuple(point) for point in arr] for arr in arr1]
            for arr1 in json_data]
    return data


def tagData(data):
    folders = ['paper', 'rock', 'scissors']
    tags = []
    for i, gesture in enumerate(data):
        tags.extend(folders[i] * len(gesture))
    return data, tags


def getTaggedData(path):
    return tagData(loadFromJson(path))


if __name__ == '__main__':
    main()
