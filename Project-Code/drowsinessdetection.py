import cv2
import numpy as np
import tensorflow as tf
from pygame import mixer

# Initialize Pygame mixer for sound
mixer.init()
sound = mixer.Sound("C:/Users/Priyadharshini/OneDrive/Desktop/AML/New folder/alarm.wav")

# Load pre-trained Haar cascades for face and eyes detection
face_cascade = cv2.CascadeClassifier("C:/Users/Priyadharshini/OneDrive/Desktop/AML/New folder/haarcascade_frontalface_alt.xml")
leye_cascade = cv2.CascadeClassifier("C:/Users/Priyadharshini/OneDrive/Desktop/AML/New folder/haarcascade_lefteye_2splits.xml")
reye_cascade = cv2.CascadeClassifier("C:/Users/Priyadharshini/OneDrive/Desktop/AML/New folder/haarcascade_righteye_2splits.xml")

# Load CNN model for eye state classification
model = tf.keras.models.load_model('cnn_model.h5')

# Initialize variables
lbl = ['Close', 'Open']
score = 0
thicc = 2

# Open the video file
video_path = "C:/Users/Priyadharshini/OneDrive/Desktop/AML/New folder/istockphoto-618279894-640_adpp_is.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    # Convert frame to grayscale for face and eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

        # Extract region of interest (ROI) for eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face ROI
        left_eye = leye_cascade.detectMultiScale(roi_gray)
        right_eye = reye_cascade.detectMultiScale(roi_gray)

        # Loop through detected left eyes
        for (lex, ley, lew, leh) in left_eye:
            l_eye = roi_color[ley:ley+leh, lex:lex+lew]
            l_eye_gray = roi_gray[ley:ley+leh, lex:lex+lew]

            # Preprocess left eye for model input
            l_eye_rgb_resized = cv2.resize(l_eye_gray, (64, 64))  # Resize to match model input size
            l_eye_rgb_resized = cv2.cvtColor(l_eye_rgb_resized, cv2.COLOR_GRAY2RGB)
            l_eye_rgb_resized = l_eye_rgb_resized.astype(np.float32) / 255.0  # Normalize pixel values
            l_eye_input = np.expand_dims(l_eye_rgb_resized, axis=0)  # Add batch dimension

            # Predict eye state using the CNN model
            lpred = np.argmax(model.predict(l_eye_input), axis=-1)[0]

            # Display eye state prediction
            if lpred == 1:
                lbl_text = 'Open'
            else:
                lbl_text = 'Closed'

            cv2.putText(frame, lbl_text, (x + lex, y + ley), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Update score based on eye state
            if lpred == 0:
                score += 1
            else:
                score -= 1

        # Evaluate and display score
        cv2.putText(frame, 'Score: ' + str(score), (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw rectangle around face and eyes
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thicc)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
