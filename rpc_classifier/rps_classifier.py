from func import *
from data import *

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
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


data, target = getTaggedData1d(path=config['data']['path'])
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, shuffle=True)

# print(X_train[:3], '\n')
# print(y_train[:10], '\n')

classifier = svm.SVC()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
