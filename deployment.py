import streamlit as st
import torch
from PIL import Image
import prediction  # Import the prediction.py file
from io import BytesIO

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model 
model = torch.load('./models/disease_classify_model.pth')
model.eval()

# List of class names
class_names = ['Caries', 'Gingivitis'] 


st.title("Image Classifier")

st.write("Upload an image for classification:")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)

    # Predict and display result

    image_path = BytesIO(uploaded_file.read()) 

    # Call the prediction function 
    prediction.pred_and_plot_image(
        model=model,
        class_names=class_names,
        image_path=image_path,
        device=device
    )
    st.pyplot()  # To display the image and the plot 
