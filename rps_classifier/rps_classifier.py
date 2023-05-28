from func import *
from data import *

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
from tensorflow import keras
import tensorflow.python.keras.layers as layers
from tensorflow.python.keras.models import save_model, load_model

import os
import configparser
import numpy as np


config = configparser.ConfigParser()
config.read('config.ini')


def main():
    path = config['data']['path']
    X_train, X_test, y_train, y_test, data, _ = prepareData(path)
    svm_model = trainSVM(X_train, X_test, y_train, y_test)
    saveSVM(svm_model, path, 'svm_classifier.pkl')
    # cnn_model = trainCNN(X_train, X_test, y_train, y_test, data)
    # saveCNN(cnn_model, path, 'cnn_classifier.h5')


# ----------------------------------data----------------------------------
def prepareData(path):
    data, target = loadTaggedData(path)
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
    num_landmarks = len(data[0])  # Number of landmarks in each sample
    X_train_reshaped = X_train.reshape(-1, num_landmarks, 1)
    X_test_reshaped = X_test.reshape(-1, num_landmarks, 1)

    cnn_classifier = keras.Sequential([
        layers.Conv1D(32, 3, activation='relu',
                      input_shape=(num_landmarks, 1)),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    cnn_classifier.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_classifier.fit(X_train_reshaped, y_train, epochs=100, batch_size=32)

    _, cnn_accuracy = cnn_classifier.evaluate(X_test_reshaped, y_test)
    print("CNN Accuracy:", cnn_accuracy)

    return cnn_classifier


# ----------------------------------save/load models----------------------------------
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


def loadSVM(path, filename):
    try:
        path_svm = os.path.join(path, filename)
        return joblib.load(path_svm)
    except Exception as e:
        print(e)
        pass


def loadCNN(path, filename):
    try:
        path_cnn = os.path.join(path, filename)
        return load_model(path_cnn)
    except Exception as e:
        print(e)
        pass


# ----------------------------------main----------------------------------
if __name__ == '__main__':
    main()
