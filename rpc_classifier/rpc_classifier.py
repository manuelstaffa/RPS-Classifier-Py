from func import *
from data import *

import time
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
    path = config['data']['path']
    data = loadFromJson(path)
    print(data)


def trainTestModel():
    return


def useModel():
    return


if __name__ == '__main__':
    main()
