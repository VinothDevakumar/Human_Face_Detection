import streamlit as st
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf

st.write("Welcome to Streamlit")

img_file_buffer = st.file_uploader('Upload a JPG image', type='jpg')
loaded_model = load_model('C:/Users/Dell/Face_Reconigition/DeepLearning/model.keras')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
classifier =loaded_model


if img_file_buffer is not None:
    fig1,fig2=st.columns(2)
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    fig1.image(img, channels="BGR")
    #st.image(img, channels="BGR")
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    plt.figure(figsize=(20,10))
    #plt.imshow(img_rgb)
    fig2.image(img_rgb)
    plt.axis('off')




