import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- Configuration ---
# Path to your saved model (relative to where app.py is run, which is the project root)
MODEL_PATH = "final_model/cat_dog_classifier_final.keras"
# Image dimensions the model expects
IMG_HEIGHT = 200
IMG_WIDTH = 200

@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_trained_model():
    """Loads the pre-trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model not found at {MODEL_PATH}. Please run main.py first to train and save the model.")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img_path):
    """Preprocesses an image for model prediction."""
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array /= 255.0 # Normalize pixel values to [0, 1]
    return img_array

def make_prediction(model, processed_img):
    """Makes a prediction using the loaded model."""
    predictions = model.predict(processed_img)
    return predictions[0] # Get probabilities for the single image

# --- Streamlit App Interface ---
st.set_page_config(page_title="Cat vs. Dog Classifier", layout="centered")

st.title("🐱🐶 Cat vs. Dog Image Classifier")
st.write("Upload an image to classify it as a cat or a dog!")

# Load the model once
model = load_trained_model()

if model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Save the uploaded file temporarily to process it
        with open("temp_uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Preprocess and predict
        processed_img = preprocess_image("temp_uploaded_image.jpg")
        predictions = make_prediction(model, processed_img)

        # Get class names from the training generator's class_indices (assuming 'cats' and 'dogs')
        # This assumes 'cats' is index 0 and 'dogs' is index 1, or vice-versa, as determined by flow_from_directory
        # For a robust app, you might save class_indices from training
        class_names = ["cat", "dog"] # Default, adjust if your train_generator.class_indices is different

        # Determine predicted class and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        predicted_class_name = class_names[predicted_class_idx]

        st.success(f"Prediction: This is a **{predicted_class_name.upper()}** with **{confidence*100:.2f}%** confidence.")
        st.write("---")
        st.subheader("Raw Probabilities:")
        st.write(f"Cat: {predictions[0]*100:.2f}%")
        st.write(f"Dog: {predictions[1]*100:.2f}%")

        # Clean up temporary file
        os.remove("temp_uploaded_image.jpg")
else:
    st.warning("Model could not be loaded. Please ensure 'main.py' has been run successfully to train and save the model.")

st.markdown("---")
st.markdown("Built with Streamlit and TensorFlow")