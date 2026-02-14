import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="EfficientNetB4 Classifier", layout="centered")

st.title("EfficientNetB4 Image Classification")

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/cnn_model.keras")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((256, 256))
    img_array = np.array(img)
    
    # Use correct EfficientNet preprocessing
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_names = [
        "CaS",
        "CoS",
        "Gum",
        "MC",
        "OC",
        "OLP",
        "OT"
        ]

    class_idx = np.argmax(preds)
    confidence = np.max(preds)
    predicted_class = class_names[class_idx]

    st.success(f"Predicted class: {predicted_class}")
    st.info(f"Confidence: {confidence:.4f}")
