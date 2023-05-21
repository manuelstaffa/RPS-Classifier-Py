from func import *
from data import *

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import save_model, load_model

import os
import configparser
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


config = configparser.ConfigParser()
config.read('config.ini')


def main():
    path = config['data']['path']
    X_train, X_test, y_train, y_test, data, target = prepareData(path)
    svm_model = trainSVM(X_train, X_test, y_train, y_test)
    saveSVM(svm_model, path, 'svm_classifier.pkl')
    cnn_model = trainCNN(X_train, X_test, y_train, y_test, data)
    saveCNN(cnn_model, path, 'cnn_classifier.h5')


# ----------------------------------data----------------------------------
def prepareData(path):
    data, target = getTaggedData1d(path)
    data, target = np.array(data), np.array(target)
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.25, shuffle=True)

    print(np.concatenate((data, target.reshape(-1, 1)), axis=1))
    return X_train, X_test, y_train, y_test, data, target


# ----------------------------------train models----------------------------------
def trainSVM(X_train, X_test, y_train, y_test):
    svm_classifier = svm.SVC()
    svm_classifier.fit(X_train, y_train)
    svm_predictions = svm_classifier.predict(X_test)

    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print("SVM Accuracy:", svm_accuracy)

    return svm_classifier


def trainCNN(X_train, X_test, y_train, y_test, data):
    cnn_classifier = Sequential()
    cnn_classifier.add(
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    cnn_classifier.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_classifier.add(Flatten())
    cnn_classifier.add(Dense(64, activation='relu'))
    cnn_classifier.add(Dense(10, activation='softmax'))

    cnn_classifier.compile(optimizer=Adam(learning_rate=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    cnn_classifier.fit(X_train, y_train, epochs=10, batch_size=32)

    _, cnn_accuracy = cnn_classifier.evaluate(X_test, y_test)
    print("CNN Accuracy:", cnn_accuracy)

    return cnn_classifier


# ----------------------------------save models----------------------------------
def saveSVM(model, path, filename):
    try:
        path_svm = os.path.join(path, filename)
        joblib.dump(model, path_svm)
    except Exception as e:
        print(e)
        pass


def saveCNN(model, path, filename):
    try:
        path_cnn = os.path.join(path, filename)
        save_model(model, path_cnn)
    except Exception as e:
        print(e)
        pass


# ----------------------------------main----------------------------------
if __name__ == '__main__':
    main()
