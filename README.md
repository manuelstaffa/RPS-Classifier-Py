# rps_classifier_py<br />
Python rock paper scissors ml classifier, made as final <br />
project of the 2023 AI/ML course @NCHU<br />

--Usage--<br />
To use the program follow these steps:<br />

    1. run data.py to convert images located in the data folder
    2. run rps_classifier.py to train and test the ml model,       
        this will save the trained ml model in the data folder
        (currently set to svm, since cnn is not saving as expected)
        (this will only work after step 1 is executed)
    3. run main.py to run main program loop
        (this will only work after step 2 is executed)

configure program behaviour in config.ini<br />

--Requirements--<br />
joblib==1.2.0<br />
mediapipe==0.9.1.0<br />
numpy==1.24.2<br />
opencv_contrib_python==4.7.0.68<br />
opencv_python==4.7.0.68<br />
scikit_learn==1.1.3<br />
tensorflow==2.11.0<br />
tensorflow_intel==2.11.0<br />
tqdm==4.65.0
