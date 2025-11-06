import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Chicken Disease Detection",
    page_icon="🐔",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Cache models to avoid reloading
@st.cache_resource
def load_models():
    """Load both HDF5 and pickle models"""
    try:
        # Load HDF5 model
        model_h5 = load_model('cnn_model.h5')
        st.success("HDF5 model loaded successfully")

        # Load pickled model
        with open('model_cnn.pkl', 'rb') as file:
            model_pkl = pickle.load(file)
        st.success("Pickled model loaded successfully")

        return model_h5, model_pkl
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def preprocess_image(img, target_size=(150, 150)):
    """
    Preprocess an image for model inference.

    Args:
        img: PIL Image object
        target_size (tuple): Target size for resizing (height, width)

    Returns:
        np.array: Preprocessed image array
    """
    # Resize image
    img = img.resize(target_size)

    # Convert to array
    img_array = image.img_to_array(img)

    # Rescale
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_disease(img, model, class_names):
    """
    Predict the disease from an image using the given model.

    Args:
        img: PIL Image object
        model: Trained model
        class_names (list): List of class names

    Returns:
        tuple: (predicted_class, confidence, probabilities)
    """
    # Preprocess image
    processed_img = preprocess_image(img)

    # Make prediction
    predictions = model.predict(processed_img)

    # Get predicted class index
    predicted_class_idx = np.argmax(predictions[0])

    # Get predicted class name
    predicted_class = class_names[predicted_class_idx]

    # Get confidence
    confidence = predictions[0][predicted_class_idx]

    return predicted_class, confidence, predictions[0]

def get_confidence_color(confidence):
    """Get color class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    # Main header
    st.markdown('<h1 class="main-header">🐔 Chicken Disease Detection System</h1>', unsafe_allow_html=True)

    # Description
    st.markdown("""
    This application uses Convolutional Neural Networks (CNN) to detect chicken diseases from fecal images.
    Upload an image of chicken feces to get instant disease classification.

    **Supported Diseases:**
    - 🐔 Coccidiosis
    - ✅ Healthy
    - 🦠 New Castle Disease
    - 🦠 Salmonella
    """)

    # Load models
    model_h5, model_pkl = load_models()

    if model_h5 is None or model_pkl is None:
        st.error("Failed to load models. Please check if 'cnn_model.h5' and 'model_cnn.pkl' files exist.")
        return

    # Class names
    class_names = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

    # File uploader
    st.markdown("## 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a chicken fecal image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of chicken feces for disease detection"
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📷 Uploaded Image")
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Uploaded Image", use_column_width=True)

        # Perform inference
        with st.spinner("Analyzing image..."):
            # Predict with HDF5 model
            pred_h5, conf_h5, probs_h5 = predict_disease(image_pil, model_h5, class_names)

            # Predict with pickled model
            pred_pkl, conf_pkl, probs_pkl = predict_disease(image_pil, model_pkl, class_names)

        with col2:
            st.markdown("### 🔍 Analysis Results")

            # HDF5 Model Results
            st.markdown("**HDF5 Model Prediction:**")
            confidence_class = get_confidence_color(conf_h5)
            st.markdown(f'<div class="prediction-box"><h3>{pred_h5}</h3><p class="{confidence_class}">Confidence: {conf_h5:.2%}</p></div>', unsafe_allow_html=True)

            # Pickled Model Results
            st.markdown("**Pickled Model Prediction:**")
            confidence_class = get_confidence_color(conf_pkl)
            st.markdown(f'<div class="prediction-box"><h3>{pred_pkl}</h3><p class="{confidence_class}">Confidence: {conf_pkl:.2%}</p></div>', unsafe_allow_html=True)

            # Check if predictions match
            if pred_h5 == pred_pkl:
                st.success("✅ Both models agree on the prediction")
            else:
                st.warning("⚠️ Models have different predictions. Consider re-training or checking the image.")

        # Detailed probabilities
        st.markdown("### 📊 Detailed Probabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**HDF5 Model Probabilities:**")
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(class_names, probs_h5, color=['red', 'green', 'orange', 'purple'])
            ax.set_xlabel('Probability')
            ax.set_title('HDF5 Model')
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("**Pickled Model Probabilities:**")
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(class_names, probs_pkl, color=['red', 'green', 'orange', 'purple'])
            ax.set_xlabel('Probability')
            ax.set_title('Pickled Model')
            plt.tight_layout()
            st.pyplot(fig)

        # Recommendations based on prediction
        st.markdown("### 💡 Recommendations")

        # Use the prediction from HDF5 model (assuming it's the primary one)
        if pred_h5 == "Healthy":
            st.success("✅ The sample appears healthy. Continue with regular monitoring.")
        else:
            st.error(f"⚠️ Potential {pred_h5} detected. Recommended actions:")
            if pred_h5 == "Coccidiosis":
                st.markdown("- Consult a veterinarian immediately")
                st.markdown("- Isolate affected birds")
                st.markdown("- Administer coccidiostats as prescribed")
                st.markdown("- Improve sanitation and reduce overcrowding")
            elif pred_h5 == "New Castle Disease":
                st.markdown("- Immediate quarantine of the flock")
                st.markdown("- Contact veterinary authorities")
                st.markdown("- Depopulation may be necessary")
                st.markdown("- Vaccination program implementation")
            elif pred_h5 == "Salmonella":
                st.markdown("- Isolate infected birds")
                st.markdown("- Antibiotic treatment under veterinary supervision")
                st.markdown("- Enhanced biosecurity measures")
                st.markdown("- Regular cleaning and disinfection")

    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This is an AI-powered diagnostic tool. Always consult with a qualified veterinarian for definitive diagnosis and treatment.")

if __name__ == "__main__":
    main()