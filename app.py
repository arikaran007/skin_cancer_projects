import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('skin_cancer_model.h5')

st.title("Skin Cancer Detection")
st.write("Upload an image for skin cancer detection.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


def preprocess_image(image):
    
    image = image.resize((512, 512))

    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)

    return image

def predict(image):

    image = preprocess_image(image)

    prediction = model.predict(image)


    return prediction

if uploaded_image is not None:

    image = Image.open(uploaded_image)
    
    prediction = predict(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    class_labels = ["Eczema", "Warts Molluscum and other Viral Infections", "Melanoma", "Atopic Dermatitis", "Basal Cell Carcinoma (BCC)", "Melanocytic Nevi (NV)", "Benign Keratosis-like Lesions (BKL)", "Psoriasis pictures Lichen Planus and related diseases", "Seborrheic Keratoses and other Benign Tumors", "Tinea Ringworm Candidiasis and other Fungal Infections"]
    predicted_class = class_labels[np.argmax(prediction)]
    st.write(f"Prediction: {predicted_class}")



