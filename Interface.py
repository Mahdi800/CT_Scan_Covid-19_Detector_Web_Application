import streamlit as st
import tensorflow as tf
#import cv2
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from io import StringIO

def footer_markdown():
    footer="""
    <style>
    a:link , a:visited{
    color: blue;
    background-color: transparent;
    text-decoration: underline;
    }
    
    a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
    }
    
    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    }
    </style>
    <div class="footer">
    <p>Developed by Mohammadmahdi Naghipour Kh.</p>
    </div>
    """
    return footer


st.title("COVID-19 Detection with CT Scans")
st.header("Using CT Scans to Detect COVID-19. To Get Started Upload Your Scans Below👇")
st.text("Please note that images are uploadable only in jpg and jpeg format.")
st.markdown(footer_markdown(),unsafe_allow_html=True)


def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (64, 64)
    image = ImageOps.fit(image, size, Image.LANCZOS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction_percentage = model.predict(data)
    prediction=prediction_percentage.round()
   
    return  prediction,prediction_percentage


uploaded_file = st.file_uploader("", type=["jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded file', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label,perc = teachable_machine_classification(image, 'model.h5')
    if label == 1:
        st.write("Negative for Covid-19. Confidence Level (from 0 to 1):",perc)
    else:
        st.write("Positive for Covid-19. Confidence Level (from 0 to 1):",1-perc)
