from func import *
from data import *

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets, metrics
import time
import json
import os
import cv2
import configparser
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


config = configparser.ConfigParser()
config.read('config.ini')


data, target = getTaggedData(path=config['data']['path'])
print("test", len(data), len(target))
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.25, shuffle=False)

classifier = svm.SVC()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
