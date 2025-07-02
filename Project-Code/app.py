from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import tensorflow as tf
from pygame import mixer
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize Pygame mixer for sound
mixer.init()
sound = mixer.Sound("C:/Users/Priyadharshini/OneDrive/Desktop/AML/New folder/alarm.wav")

# Load pre-trained Haar cascades for face and eyes detection
face_cascade = cv2.CascadeClassifier("C:/Users/Priyadharshini/OneDrive/Desktop/AML/New folder/haarcascade_frontalface_alt.xml")
leye_cascade = cv2.CascadeClassifier("C:/Users/Priyadharshini/OneDrive/Desktop/AML/New folder/haarcascade_lefteye_2splits.xml")
reye_cascade = cv2.CascadeClassifier("C:/Users/Priyadharshini/OneDrive/Desktop/AML/New folder/haarcascade_righteye_2splits.xml")

# Load CNN model for eye state classification
model = tf.keras.models.load_model('C:/Users/Priyadharshini/OneDrive/Desktop/AML/New folder/cnn_model.h5')

lbl = ['Close', 'Open']
score = 0
thicc = 2

video_path = ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            global video_path
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)
            return redirect(url_for('play'))
    return render_template('upload.html')

@app.route('/play')
def play():
    return render_template('play.html')

def generate_frames():
    global score
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            left_eye = leye_cascade.detectMultiScale(roi_gray)
            right_eye = reye_cascade.detectMultiScale(roi_gray)

            for (lex, ley, lew, leh) in left_eye:
                l_eye = roi_color[ley:ley+leh, lex:lex+lew]
                l_eye_gray = roi_gray[ley:ley+leh, lex:lex+lew]
                l_eye_rgb_resized = cv2.resize(l_eye_gray, (64, 64))
                l_eye_rgb_resized = cv2.cvtColor(l_eye_rgb_resized, cv2.COLOR_GRAY2RGB)
                l_eye_rgb_resized = l_eye_rgb_resized.astype(np.float32) / 255.0
                l_eye_input = np.expand_dims(l_eye_rgb_resized, axis=0)
                lpred = np.argmax(model.predict(l_eye_input), axis=-1)[0]

                lbl_text = 'Open' if lpred == 1 else 'Closed'
                cv2.putText(frame, lbl_text, (x + lex, y + ley), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if lpred == 0:
                    score += 1
                else:
                    score -= 1

            cv2.putText(frame, 'Score: ' + str(score), (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thicc)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
