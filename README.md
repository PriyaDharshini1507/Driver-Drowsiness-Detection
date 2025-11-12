# Driver Drowsiness Detection Using Eye

## Overview
This project implements a real-time driver drowsiness detection system using computer vision and deep learning. The system analyzes the driver’s eye state (open or closed) from a video feed and triggers an alarm if signs of drowsiness are detected. It leverages Haar cascade classifiers for face and eye detection, along with a pre-trained Convolutional Neural Network (CNN) model for eye state classification. The alert mechanism aims to enhance road safety by preventing accidents caused by driver fatigue.

---

## Problem Statement
The objective of this project is to develop an automated system that detects driver drowsiness through continuous monitoring of eye movements. Using a webcam or video input, the system identifies the face and eyes in each frame, classifies the eye state using a trained CNN model, and calculates a score based on how long the eyes remain closed. When the score exceeds a defined threshold, an alarm sound is triggered to alert the driver.

---

## System Workflow

1. **Video Frame Capture:**  
   The system reads frames continuously from a webcam or video file.

2. **Face and Eye Detection:**  
   - Haar cascade classifiers are used to detect the face and eyes in each frame.  
   - Separate classifiers are applied for detecting the left and right eyes.

3. **Eye State Classification:**  
   - Each detected eye region is preprocessed and passed to a CNN model (`cnn_model.h5`) for classification.  
   - The model predicts whether the eye is open or closed.

4. **Drowsiness Detection Logic:**  
   - A score is incremented if both eyes remain closed over consecutive frames.  
   - When the score surpasses the defined limit, an alert (`alarm.wav`) is played using the Pygame mixer.

5. **Visualization:**  
   - The live video feed displays the detected face, eye state, and current score in real time.

---

## Technologies and Libraries

| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | Python 3.x |
| Deep Learning Framework | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Audio Alert | Pygame |
| Model | Pre-trained CNN (`cnn_model.h5`) |
| Classifiers | Haar Cascade XML files for face and eyes |

---

## How It Works

1. The system initializes the Haar cascade classifiers and the pre-trained CNN model.  
2. Frames are captured from a video file or webcam.  
3. The face is detected and regions containing eyes are extracted.  
4. The CNN model classifies each eye as either open or closed.  
5. A score variable tracks the duration of eye closure.  
6. If the eyes remain closed beyond the set threshold, an audio alert (`alarm.wav`) is triggered.

---

## Requirements

1. Python 3.8 or higher
2. TensorFlow
3. Keras
4. OpenCV


