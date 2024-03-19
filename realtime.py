import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the HDF5 model file
@st.cache_data(allow_output_mutation=True)
def load_hdf5_model():
    model = load_model('model.h5')  # Replace 'your_model.h5' with the path to your HDF5 model file
    return model

model = load_hdf5_model()

# Function to perform image detection
def detect_objects(image):
    # Preprocess the image
    resized_image = cv2.resize(image, (224, 224))
    resized_image = resized_image / 255.0  # Normalize pixel values
    resized_image = np.expand_dims(resized_image, axis=0)  # Add batch dimension

    # Perform inference
    predictions = model.predict(resized_image)

    return predictions

# Streamlit app
def main():
    st.title('Image Detection App')

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Perform object detection
        predictions = detect_objects(image)

        # Display results
        st.write("Predictions:", predictions)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

if __name__ == '__main__':
    main()
