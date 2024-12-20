import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
from typing import List, Tuple

def pred_and_plot_images_streamlit(
    model: nn.Module,
    class_names: List[str],
    image_size: Tuple[int, int] = (224, 224),
    transform: transforms.Compose = None,
    device: torch.device = torch.device("cpu")
):
    # Create a transformation if none is provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    # Ensure the model is on the target device
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    # Streamlit file uploader
    uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Open and transform the image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Transform image for prediction
            transformed_image = transform(img).unsqueeze(dim=0)  # Add batch dimension

            # Make prediction
            with torch.inference_mode():
                pred_logits = model(transformed_image.to(device))
                pred_probs = torch.softmax(pred_logits, dim=1)
                pred_label = torch.argmax(pred_probs, dim=1).item()

            # Display the prediction and probability
            st.write(f"Prediction: {class_names[pred_label]}")
            st.write(f"Probability: {pred_probs[0, pred_label].item():.3f}")
