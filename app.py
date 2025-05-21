import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model (SavedModel format folder)
model = tf.keras.models.load_model('weather_classifier')

# Define the class names in the order used during training
class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

st.title("🌦️ Weather Image Classifier")
st.markdown("Upload a weather image, and the model will predict the weather condition with high accuracy.")

# File uploader widget
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert image to array and normalize
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        st.write("Input shape to model:", img_array.shape)
        st.write("Running prediction...")

        # Use the serving signature for prediction
        infer = model.signatures["serving_default"]
        input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        output = infer(input_tensor)
        prediction = list(output.values())[0].numpy()

        st.write("Raw prediction output:", prediction)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)")

    except Exception as e:
        st.error(f"❌ Prediction failed:\n\n{e}")
