import streamlit as st
import numpy as np
from PIL import Image

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️"
)

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and AI will try to recognize it.")

# --------------------------------------------------
# Load and train model (cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    digits = load_digits()
    X = digits.images.reshape(len(digits.images), -1) / 16.0
    y = digits.target

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


model = load_model()
st.success("Model loaded successfully!")

# --------------------------------------------------
# File uploader
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        # Show image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Image preprocessing
        img_gray = image.convert("L")
        img_resized = img_gray.resize((8, 8))

        img_array = np.array(img_resized)
        img_array = img_array / 16.0
        img_flat = img_array.flatten().reshape(1, -1)

        # Prediction
        prediction = model.predict(img_flat)[0]
        probabilities = model.predict_proba(img_flat)[0]

        st.subheader(f"Predicted digit: **{prediction}**")

        st.write("### Probabilities:")
        for i, prob in enumerate(probabilities):
            st.write(f"Digit {i}: {prob:.2%}")

    except Exception as e:
        st.error(f"Error processing image: {e}")

# --------------------------------------------------
# Sidebar instructions
# --------------------------------------------------
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload an image of a handwritten digit (0–9)
2. The image will be resized to 8×8 pixels
3. AI model will predict the digit

**For best results:**
- White background  
- Black digit  
- Centered digit  
- Minimal noise
""")
