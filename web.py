import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os

def model_prediction(test_image):
    model = tf.keras.models.load_model("Train_potato_saved.keras")
    image = tf.keras.utils.load_img(test_image, target_size=(128, 128))
    array = tf.keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)
    prediction = model.predict(array)
    return np.argmax(prediction)


# Streamlit UI Design
st.set_page_config(page_title="Plant Disease Detection", layout="wide")
st.sidebar.title("🌱 Plant Disease Detection System")
app_mode = st.sidebar.radio("Select Page", ["🏠 Home", "🔍 Disease Recognition"])

# Display Header Image
st.image("potato.jpg",use_column_width=True)

if app_mode == "🏠 Home":
    st.markdown(
    """
    <h1 style='text-align: center; color: green;'>Plant Disease Detection System</h1>
    <h3 style='text-align: center;'>Enhancing Sustainable Agriculture</h3>
    <p style='text-align: justify;'>This application uses deep learning to detect plant diseases from images.
    Simply upload an image of a potato leaf, and the model will classify it as
    either healthy or affected by Early/Late Blight.</p>
    <p style='text-align: center;'><b>Let's contribute to healthier plants and sustainable farming!</b></p>
    """, 
    unsafe_allow_html=True
)
    
elif app_mode == "🔍 Disease Recognition":
    st.header("🔍 Upload an Image for Disease Recognition")
    test_image = st.file_uploader("📂 Choose an image", type=["jpg", "png", "jpeg"])
    
    if test_image:
        st.image(test_image, caption="Uploaded Image",width = 200)
        if st.button("🔎 Predict Disease"):
            with st.spinner("Processing Image..."):
                result_index = model_prediction(test_image)
                class_names = ['Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy']
                prediction_text = class_names[result_index]
                st.success(f'✅ The model predicts: **{prediction_text}**')
                st.snow()
