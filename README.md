# Badminton-Shots-Recognition-and-Analysis

This Project will detect the first player keypoints and detect the shots played between the rally.

I have detected the keypoints using Yolov7 and extracted those keypoints and calculates the angle between joints which are crucial to play various shots between the rally. You can see these implementation in the keypointsfinder.py file.

Used those extracted points and angles to train the model using LSTM.

Detected shots payed by the player in the file badminton_shot_identification_and_counts.py file.

You can see the results in Drop.mp4 clip.
