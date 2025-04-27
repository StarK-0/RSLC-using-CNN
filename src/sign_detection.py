import cv2
import numpy as np
import tensorflow as tf
import os

model = tf.keras.models.load_model("../models/sign_model.h5")
labels = os.listdir("E:/Final Year Project/Final Review and Report/signlanguage/dataset")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (64, 64))
    normalized_frame = np.expand_dims(resized_frame / 255.0, axis=0)

    prediction = model.predict(normalized_frame)
    predicted_label = labels[np.argmax(prediction)]

    cv2.putText(frame, predicted_label, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
