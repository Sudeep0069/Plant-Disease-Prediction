import os
import json
import numpy as np
import tensorflow as tf
import gradio as gr

# Load Keras model and class names
MODEL_PATH = "plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found in the workspace.")
    exit(1)

if not os.path.exists(CLASS_NAMES_PATH):
    print(f"Error: Class names JSON '{CLASS_NAMES_PATH}' not found in the workspace.")
    exit(1)

print(f"Loading model from '{MODEL_PATH}' and class names...")
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)

# Disease Info and Remedies dictionaries (extracted from the notebook)
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

# Normalize lookup keys (removing underscores, spaces, hyphens) to prevent KeyErrors
def get_dict_val(dictionary, label):
    def clean(s):
        return "".join(c.lower() for c in s if c.isalnum())
    
    label_clean = clean(label)
    for key, val in dictionary.items():
        if clean(key) == label_clean:
            return val
    return "No details/remedies available for this class."

def predict(image):
    # Resize and preprocess image
    image = image.resize(IMG_SIZE)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)

    predicted_idx = np.argmax(predictions)
    predicted_label = class_names[predicted_idx]
    confidence = float(np.max(predictions))

    info = get_dict_val(disease_info, predicted_label)
    remedy = get_dict_val(disease_remedies, predicted_label)

    return (
        {predicted_label: confidence},
        info,
        remedy
    )

# Setup Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Textbox(label="Disease Information"),
        gr.Textbox(label="Recommended Countermeasures")
    ],
    title="Plant Disease Prediction Web App",
    description="Upload an image of a plant leaf (Tomato, Potato, or Pepper) to identify potential disease and get countermeasures."
)

if __name__ == "__main__":
    print("Starting Gradio Web Application...")
    # Share=True creates a public link if needed, but we run locally by default
    interface.launch(share=True)
