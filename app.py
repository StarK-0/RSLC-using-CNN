import cv2
import numpy as np
import tensorflow as tf
import os
import streamlit as st
import time

# Load model
model_path = "models/sign_model.h5"
if os.path.exists("models/sign_model_final.h5"):
    model_path = "models/sign_model_final.h5"  # Use fine-tuned model if available

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# Load class names
if os.path.exists("models/class_names.txt"):
    with open("models/class_names.txt", 'r') as f:
        labels = [line.strip() for line in f.readlines()]
else:
    # Fallback to directory names
    labels = sorted(os.listdir("D:/signlanguage/dataset"))

st.title("Sign Language Detector")
st.write("Hold your hand in the frame to detect sign language gestures")

# Image preprocessing
def preprocess_image(img):
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img_resized = cv2.resize(img_rgb, (128, 128))
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    return np.expand_dims(img_normalized, axis=0)

# Prediction with confidence
def predict_with_confidence(frame):
    # Create a region of interest in the center of the frame
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    roi_size = min(w, h) // 2
    
    # Extract ROI (region of interest)
    roi = frame[
        max(0, center_y - roi_size):min(h, center_y + roi_size),
        max(0, center_x - roi_size):min(w, center_x + roi_size)
    ]
    
    if roi.size == 0:  # Check if ROI is empty
        return "No hand detected", 0
    
    # Preprocess
    processed_img = preprocess_image(roi)
    
    # Prediction
    prediction = model.predict(processed_img)[0]
    
    # Get top prediction and confidence
    top_idx = np.argmax(prediction)
    confidence = prediction[top_idx] * 100
    
    # Return label and confidence
    return labels[top_idx], confidence

# Settings sidebar
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (%)", 
    min_value=0, 
    max_value=100, 
    value=50
)

show_roi = st.sidebar.checkbox("Show Region of Interest", value=True)
prediction_delay = st.sidebar.slider(
    "Prediction Delay (ms)", 
    min_value=100, 
    max_value=2000, 
    value=500
)

# Start camera
if st.button("Start Camera"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    st_text = st.empty()
    
    last_prediction_time = time.time() * 1000
    last_prediction = "Waiting..."
    last_confidence = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from camera")
            break
        
        # Draw ROI rectangle
        if show_roi:
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            roi_size = min(w, h) // 2
            cv2.rectangle(
                frame,
                (center_x - roi_size, center_y - roi_size),
                (center_x + roi_size, center_y + roi_size),
                (0, 255, 0), 2
            )
        
        # Update prediction every X milliseconds
        current_time = time.time() * 1000
        if current_time - last_prediction_time > prediction_delay:
            prediction, confidence = predict_with_confidence(frame)
            
            if confidence >= confidence_threshold:
                last_prediction = prediction
                last_confidence = confidence
            else:
                last_prediction = "Low confidence"
                last_confidence = confidence
                
            last_prediction_time = current_time
        
        # Display prediction on frame
        cv2.putText(
            frame,
            f"{last_prediction} ({last_confidence:.1f}%)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        
        # Show the frame in Streamlit
        stframe.image(frame, channels="BGR")
        st_text.markdown(f"### Detected Sign: **{last_prediction}** with {last_confidence:.1f}% confidence")
        
        # Break if the window is closed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()