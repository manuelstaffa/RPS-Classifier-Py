from func import *
from data import *

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.externals import joblib
from tensorflow import keras
import tensorflow.python.keras.layers as layers

import os
import configparser
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# load config
config = configparser.ConfigParser()
config.read('config.ini')
path = config['data']['path']


# get data
data, target = getTaggedData1d(path=config['data']['path'])
data, target = np.array(data), np.array(target)
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.25, shuffle=True)

print(np.concatenate((data, target.reshape(-1, 1)), axis=1))


# SVM classifier
svm_classifier = svm.SVC()
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)


# CNN classifier
num_landmarks = len(data[0])  # Number of landmarks in each sample
X_train_reshaped = X_train.reshape(-1, num_landmarks, 1)
X_test_reshaped = X_test.reshape(-1, num_landmarks, 1)

cnn_classifier = keras.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(num_landmarks, 1)),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])
cnn_classifier.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_classifier.fit(X_train_reshaped, y_train, epochs=100, batch_size=32)
_, cnn_accuracy = cnn_classifier.evaluate(X_test_reshaped, y_test)


# print results
print("SVM Accuracy:", svm_accuracy)
print("CNN Accuracy:", cnn_accuracy)


# save models
path_svm = path = os.path.join(path, 'svm_classifier.pkl')
joblib.dump(svm_classifier, path_svm)

path_cnn = path = os.path.join(path, 'cnn_classifier.pkl')
joblib.dump(cnn_classifier, path_svm)
