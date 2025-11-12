# 🚗 Driver Drowsiness Detection Using Eye 👁️  

## 🧠 Overview  
Driver fatigue is one of the leading causes of road accidents worldwide. This project presents a **real-time driver drowsiness detection system** that monitors a driver’s eye state using **computer vision** and **deep learning** techniques. The system issues an alert when it detects prolonged eye closure, helping prevent accidents caused by drowsiness or fatigue.

---

## 🎯 Problem Statement  
The objective of this project is to develop a **real-time system** capable of detecting driver drowsiness through eye state analysis. The system continuously captures video frames using a webcam, detects the driver’s face and eyes, and classifies the eye state as **open** or **closed** using a **Convolutional Neural Network (CNN)** model.  
If the eyes remain closed beyond a certain duration, the system triggers an **audio alert** to wake up the driver and ensure road safety.

---

## ⚙️ System Workflow  

1. **Frame Capture:**  
   The webcam captures continuous frames of the driver.  

2. **Face and Eye Detection:**  
   - The **Haar Cascade classifier** detects the face and extracts a Region of Interest (ROI).  
   - Separate classifiers detect the **left** and **right eyes** within the ROI.  

3. **Eye State Classification:**  
   - Each eye image is pre-processed and passed to a **pre-trained CNN model (cnnCat2.h5)**.  
   - The model classifies the eye as **open** or **closed**.  

4. **Drowsiness Scoring:**  
   - A **score counter** increments when both eyes are closed.  
   - If the score exceeds a defined **threshold**, an **alert sound (alarm.wav)** is triggered.  

---

## 🧩 Technologies and Libraries  

| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | Python 3.x |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Audio Alert | Pygame |
| Model | Pre-trained CNN (cnnCat2.h5) |
| Classifiers | Haar Cascade XML files for face and eyes |

---

## 🗂️ Project Structure  

Driver-Drowsiness-Detection/
│
├── haarcascade_frontalface_alt.xml
├── haarcascade_lefteye_2splits.xml
├── haarcascade_righteye_2splits.xml
├── cnnCat2.h5
├── alarm.wav
├── drowsiness_detection.py
├── README.md
└── requirements.txt


## 🧾 Requirements

  1. Python 3.8 or higher
  
  2. TensorFlow / Keras
  
  3. OpenCV
