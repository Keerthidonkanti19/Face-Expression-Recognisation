import cv2
from keras.models import model_from_json
import numpy as np
import pyttsx3
import time
from collections import deque, Counter

# Initialize the speech engine
engine = pyttsx3.init()
last_spoken = ""
last_spoken_time = 0
cooldown_secs = 3  # Minimum delay between speeches (in seconds)

# Load model
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.weights.h5")

# Load face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Preprocess input image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion labels
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# History buffer to stabilize predictions
prediction_history = deque(maxlen=5)

# Start webcam
webcam = cv2.VideoCapture(0)

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray_eq, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            image = gray_eq[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)

            # Confidence threshold
            confidence = np.max(pred)
            if confidence < 0.6:
                continue

            # Most likely emotion
            emotion = labels[pred.argmax()]
            prediction_history.append(emotion)
            stable_emotion = Counter(prediction_history).most_common(1)[0][0]

            # Determine direction
            frame_width = im.shape[1]
            face_center = p + r / 2
            if face_center < frame_width / 3:
                direction = "left"
            elif face_center > frame_width * 2 / 3:
                direction = "right"
            else:
                direction = "center"

            spoken_text = f"{stable_emotion} on the {direction}"
            current_time = time.time()

            if spoken_text != last_spoken or (current_time - last_spoken_time > cooldown_secs):
                engine.say(spoken_text)
                engine.runAndWait()
                last_spoken = spoken_text
                last_spoken_time = current_time

            # Display emotion label
            cv2.putText(im, stable_emotion, (p - 10, q - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Output", im)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()

