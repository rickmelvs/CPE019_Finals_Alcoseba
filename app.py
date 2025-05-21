import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load trained model
model = load_model('weather_classifier.keras')

# Define class names (order must match model training)
class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

st.title("🌦️ Weather Image Classifier")
st.markdown("Upload a weather image, and the model will predict the weather condition with high accuracy.")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)")
