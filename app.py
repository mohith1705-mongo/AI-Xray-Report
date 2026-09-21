import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI X-Ray Report Generator",
    page_icon="🩻",
    layout="centered"
)


# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            "xray_resnet18.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model()


# -----------------------------
# Image Preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Report Generator
# -----------------------------

def generate_report(predicted_class, confidence):

    confidence = confidence * 100

    if predicted_class == "PNEUMONIA":

        report = f"""
### AI CHEST X-RAY REPORT

**Finding:**  
The AI model classified this X-ray as **PNEUMONIA**.

**Model Confidence:**  
{confidence:.2f}%

**AI Summary:**  
The image was classified into the pneumonia category
by the trained AI model.

**Important Note:**  
This is an educational AI prototype and is not a
medical diagnosis. Review by a qualified radiologist
is required.
"""

    else:

        report = f"""
### AI CHEST X-RAY REPORT

**Finding:**  
The AI model classified this X-ray as **NORMAL**.

**Model Confidence:**  
{confidence:.2f}%

**AI Summary:**  
The image was classified into the normal category
by the trained AI model.

**Important Note:**  
This is an educational AI prototype and is not a
medical diagnosis. Review by a qualified radiologist
is required.
"""

    return report


# -----------------------------
# Streamlit Interface
# -----------------------------

st.title("🩻 AI X-Ray Report Generator")

st.write(
    "Upload a chest X-ray image to analyze it "
    "using the trained AI model."
)

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Analyze X-Ray
# -----------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded X-Ray")

    st.image(
        image,
        caption="Chest X-Ray",
        use_container_width=True
    )

    if st.button("🔍 Analyze X-Ray"):

        image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():

            output = model(image_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

        class_names = [
            "NORMAL",
            "PNEUMONIA"
        ]

        predicted_label = class_names[
            predicted_class
        ]

        confidence = probabilities[
            0,
            predicted_class
        ].item()

        # Prediction
        st.subheader("AI Prediction")

        st.success(
            f"{predicted_label} "
            f"({confidence * 100:.2f}% confidence)"
        )

        # Report
        st.markdown(
            generate_report(
                predicted_label,
                confidence
            )
        )