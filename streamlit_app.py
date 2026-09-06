import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")

MODEL_PATH = "plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_assets()
IMG_SIZE = (224, 224)

disease_info = {
    "Pepper__bell___Bacterial_spot": "A bacterial disease causing dark lesions on pepper leaves and fruits.",
    "Pepper__bell___healthy": "The pepper leaf appears healthy with no signs of disease.",
    "Potato___Early_blight": "A fungal disease caused by Alternaria solani, characterized by concentric rings on older leaves.",
    "Potato___Late_blight": "A destructive disease caused by Phytophthora infestans that can rapidly spread under humid conditions.",
    "Potato___healthy": "The potato leaf appears healthy with no visible disease symptoms.",
    "Tomato___Bacterial_spot": "A bacterial infection that produces small dark spots on leaves and fruits.",
    "Tomato___Early_blight": "A fungal disease affecting older tomato leaves, causing target-like lesions.",
    "Tomato___Late_blight": "A serious disease caused by Phytophthora infestans that affects leaves, stems, and fruits.",
    "Tomato___Leaf_Mold": "A fungal disease favored by high humidity, producing yellow patches on leaves.",
    "Tomato___Septoria_leaf_spot": "A fungal disease causing numerous small circular spots with dark margins.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "An infestation by tiny pests that feed on plant sap, causing yellow speckling.",
    "Tomato___Target_Spot": "A fungal disease characterized by brown lesions with concentric rings.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "A viral disease transmitted by whiteflies that causes yellowing and leaf curling.",
    "Tomato___Tomato_mosaic_virus": "A viral disease causing mottled patterns and distortion of tomato leaves.",
    "Tomato___healthy": "The tomato leaf appears healthy with no visible disease symptoms."
}

disease_remedies = {
    "Pepper__bell___Bacterial_spot": "Remove infected leaves, avoid overhead irrigation, and apply copper-based bactericides.",
    "Pepper__bell___healthy": "No disease detected. Continue proper watering and nutrient management.",
    "Potato___Early_blight": "Remove infected foliage, rotate crops, and apply fungicides containing chlorothalonil or mancozeb.",
    "Potato___Late_blight": "Destroy infected plants immediately and apply recommended fungicides to prevent spread.",
    "Potato___healthy": "No disease detected. Maintain regular crop monitoring.",
    "Tomato___Bacterial_spot": "Use disease-free seeds, avoid working with wet plants, and apply copper-based sprays.",
    "Tomato___Early_blight": "Prune affected leaves, rotate crops, and use appropriate fungicides.",
    "Tomato___Late_blight": "Remove infected plants promptly and apply preventive fungicides.",
    "Tomato___Leaf_Mold": "Reduce greenhouse humidity, improve ventilation, and remove infected leaves.",
    "Tomato___Septoria_leaf_spot": "Remove infected foliage, mulch around plants, and use approved fungicides.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use insecticidal soap or miticides and encourage beneficial predators.",
    "Tomato___Target_Spot": "Improve airflow around plants and apply fungicides if necessary.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies using traps or insecticides and remove infected plants.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants and disinfect gardening tools to prevent transmission.",
    "Tomato___healthy": "No disease detected. Continue standard crop management practices."
}

def clean(s):
    return "".join(c.lower() for c in s if c.isalnum())

def get_val(dictionary, label):
    clean_lbl = clean(label)
    for k, v in dictionary.items():
        if clean(k) == clean_lbl:
            return v
    return "Information not available."

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image (Tomato, Potato, Pepper) to diagnose plant health and recommended actions.")

uploaded_file = st.file_uploader("Upload leaf photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Target Leaf", width=300)

    with st.spinner("Analyzing image..."):
        resized_img = img.resize(IMG_SIZE)
        img_array = np.array(resized_img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)
        idx = int(np.argmax(preds))
        label = class_names[idx]
        conf = float(np.max(preds)) * 100

        st.success(f"**Diagnosis:** {label} ({conf:.2f}% confidence)")
        st.info(f"**Disease Info:** {get_val(disease_info, label)}")
        st.warning(f"**Action Plan:** {get_val(disease_remedies, label)}")