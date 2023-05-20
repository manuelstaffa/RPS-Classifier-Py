from func import *
from data import *

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras
import tensorflow.python.keras.layers as layers

import configparser
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


config = configparser.ConfigParser()
config.read('config.ini')


data, target = getTaggedData1d(path=config['data']['path'])
data, target = np.array(data), np.array(target)
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, shuffle=True)

# print(X_train[:3], '\n')
# print(y_train[:10], '\n')

# SVM Classifier
svm_classifier = svm.SVC()
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)

# CNN Classifier
num_landmarks = len(data[0])  # Number of landmarks in each sample
X_train_reshaped = X_train.reshape(-1, num_landmarks, 1)
X_test_reshaped = X_test.reshape(-1, num_landmarks, 1)

cnn_model = keras.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(num_landmarks, 1)),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32)
_, cnn_accuracy = cnn_model.evaluate(X_test_reshaped, y_test)
print("CNN Accuracy:", cnn_accuracy)
