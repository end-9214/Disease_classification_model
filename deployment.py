import streamlit as st
import torch
import torchvision.models as models
from torch import nn
import prediction

# Load your trained model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the model architecture (adjust based on your trained model)
weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
output_shape = 2  
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=output_shape, bias=True)
)

# Load the trained model weights
model.load_state_dict(torch.load('./models/disease_classify_model.pth'))
model.eval()  # Set the model to evaluation mode
model.to(device)

# List of class names (update this based on your dataset)
class_names = ['Caries', 'Gingivitis']

# Call the prediction function
st.title("Disease Classifier")

# Call the modified prediction function with Streamlit
prediction.pred_and_plot_images_streamlit(model, class_names, device=device)
