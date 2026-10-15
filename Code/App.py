import streamlit as st
import numpy as np
import pandas as pd
import random
from keras.models import load_model

# ===========================
# CONFIGURATION & STYLING
# ===========================
st.set_page_config(
    page_title="Arabic Handwritten Character Recognition",
    page_icon="📝",
    layout="wide"
)

# Custom light theme CSS
st.markdown("""
<style>
/* Main background */
body {
    background-color: #f6f9fc;
}

/* Center the title */
.title {
    text-align: center;
    font-size: 40px !important;
    font-weight: 800;
    color: #2c3e50;
}

/* Subtext */
.subtitle {
    text-align: center;
    font-size: 18px !important;
    color: #6c757d;
}

/* Card container */
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Prediction labels */
.pred-label {
    font-size: 22px !important;
    font-weight: 700;
    color: #2980b9;
}

.true-label {
    font-size: 20px !important;
    font-weight: 600;
    color: #27ae60;
}

/* Hide Streamlit default slider label spacing */
.css-1cypcdb {
    padding-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# PATHS (CHANGE THESE IF NEEDED)
# ===========================
CSV_IMAGES_PATH = r"csvTestImages 3360x1024.csv"
CSV_LABELS_PATH = r"csvTestLabel 3360x1.csv"
MODEL_PATH      = "arabic_handwriting_cnn.h5"

# ===========================
# ARABIC LETTERS MAPPING (0–27)
# ===========================
arabic_letters = [
    "ا","ب","ت","ث","ج","ح","خ","د","ذ","ر","ز","س","ش","ص",
    "ض","ط","ظ","ع","غ","ف","ق","ك","ل","م","ن","ه","و","ي"
]

# ===========================
# LOADING FUNCTIONS
# ===========================
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

@st.cache_data
def load_data():
    images = pd.read_csv(CSV_IMAGES_PATH, header=None).values
    labels = pd.read_csv(CSV_LABELS_PATH, header=None).values

    # Reshape & normalize exactly like in the notebook
    images = images.reshape(-1, 32, 32, 1) / 255.0

    # labels are 1–28 in the CSV → convert to 0–27
    labels = labels.squeeze().astype("int32") - 1

    return images, labels

# ===========================
# UI HEADER
# ===========================
st.markdown("<h1 class='title'>Arabic Handwritten Character Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Test a CNN model trained on the Arabic Handwritten Characters Dataset (AHCD)</p>", unsafe_allow_html=True)
st.write("")

model = load_cnn_model()
all_images, all_labels = load_data()

# ===========================
# CENTERED SLIDER + BUTTON
# ===========================
st.markdown("### Choose how many random characters to display")
colA, colB, colC = st.columns([1, 2, 1])

with colB:
    num_samples = st.slider(
        label=" ",
        min_value=1,
        max_value=10,
        value=3
    )

st.write("")  # spacing

btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    generate = st.button("Generate Predictions")

st.write("")  # spacing

# ===========================
# PREDICTION DISPLAY
# ===========================
if generate:
    indices = random.sample(range(len(all_images)), num_samples)

    for i in indices:
        img = all_images[i]
        true_idx = all_labels[i]

        # Predict
        prediction = model.predict(img[np.newaxis, ...], verbose=0)
        pred_idx = int(np.argmax(prediction))

        # Card layout
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 3])

        with col1:
            st.image(
                img.squeeze(),
                width=140,
                clamp=True,
            )

        with col2:
            st.markdown(
                f"<p class='pred-label'>Predicted: {arabic_letters[pred_idx]}</p>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p class='true-label'>True: {arabic_letters[true_idx]}</p>",
                unsafe_allow_html=True
            )

            # Show top 3 probabilities
            probs = prediction[0]
            top3 = probs.argsort()[-3:][::-1]
            st.write("Top predictions:")
            for j in top3:
                st.write(f"- **{arabic_letters[j]}** — {probs[j]*100:.1f}%")

        st.markdown("</div>", unsafe_allow_html=True)
