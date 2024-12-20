import streamlit as st
import torch
from PIL import Image
import prediction  # Import the prediction.py file
from io import BytesIO

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model (ensure you load the trained model in the correct way)
# model = torch.load('model.pth')  # Example to load your model
model = torch.load('./models/disease_classify_model.pth')
model.eval()

# List of class names (make sure these match your model's output classes)
class_names = ['Caries', 'Gingivitis'] 

# Streamlit UI

st.title("Image Classifier")

st.write("Upload an image for classification:")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)

    # Predict and display result
    # Use the pred_and_plot_image function from your prediction.py
    image_path = BytesIO(uploaded_file.read())  # Convert uploaded file to a readable format

    # Call the prediction function from your 'prediction.py'
    prediction.pred_and_plot_image(
        model=model,
        class_names=class_names,
        image_path=image_path,
        device=device
    )
    st.pyplot()  # To display the image and the plot in the Streamlit app
