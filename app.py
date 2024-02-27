import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image

# Load the model and tokenizer
processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")

# Title and instructions
st.title("Deepfake Image Detection")
st.write("Upload an image to check if it's likely real or fake.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Preprocess the image
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Display image and prediction
    st.image(image, caption="Uploaded Image", width=300)

    if predicted_class == 0:
        st.write("**Prediction:** Real")
    else:
        st.write("**Prediction:** Deepfake")

else:
    st.write("Please upload an image to make a prediction.")