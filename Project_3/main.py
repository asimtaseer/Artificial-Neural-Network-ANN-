import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("my_model.h5")

# Class labels
class_names = ['COVID', 'Lung_Opacity', 'Normal']

# Function to predict image class
def predict_image_class(image):
    img = image.resize((150, 150))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    return class_names[predicted_class], confidence

# Streamlit UI
st.title("🩺 COVID-19 Detection Using Deep Learning")
st.write("Upload a chest X-ray image to classify it as **COVID**, **Lung Opacity**, or **Normal**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        predicted_class, confidence = predict_image_class(image)
        st.success(f"**Predicted Class:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}")


