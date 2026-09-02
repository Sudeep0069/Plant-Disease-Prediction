# 🌿 Plant Disease Prediction using CNN

> A deep learning application that identifies plant diseases from leaf
> images using **transfer learning with MobileNetV2**, with a **Gradio
> web interface** for real-time image-based prediction.

```
<img width="1919" height="862" alt="interface" src="https://github.com/user-attachments/assets/8b7aec58-350f-48d2-a969-98a9309da536" />

```
## 📌 Project Overview

Plant diseases can significantly affect crop health and productivity.
This project uses a convolutional neural network approach to classify
plant leaf images into **15 disease/healthy classes**.

The model is built with **TensorFlow/Keras** using **MobileNetV2
pretrained on ImageNet**. Transfer learning is combined with image
augmentation, class weighting, and fine-tuning to improve performance
across an imbalanced dataset.

The trained model is then integrated into a **Gradio interface**,
allowing users to upload a leaf image and receive:

-   🔎 Predicted disease/class
-   📊 Prediction confidence
-   📝 Disease information
-   🛠️ Recommended countermeasures

------------------------------------------------------------------------

## ✨ Key Features

-   🧠 CNN-based image classification using **MobileNetV2**
-   🔄 Transfer learning with ImageNet pretrained weights
-   🖼️ Input images resized to **224 × 224**
-   🎛️ Data augmentation using:
    -   Random horizontal & vertical flips
    -   Random rotation
    -   Random zoom
-   ⚖️ Class weighting to address class imbalance
-   🎯 Early stopping and best-model checkpointing
-   🔧 Fine-tuning of the pretrained backbone
-   📈 Training/validation performance monitoring
-   📊 Classification report and confusion matrix
-   🌐 Interactive Gradio prediction interface
-   💡 Disease-specific information and countermeasures

------------------------------------------------------------------------

## 📂 Dataset

The project uses the **PlantVillage** dataset.

The notebook contains **15 classes** covering pepper, potato, and tomato
leaves.

### Dataset statistics

  Metric                                  Value
  ----------------------------- ---------------
  Number of classes                      **15**
  Images loaded by TensorFlow        **20,638**
  Training images                    **16,511**
  Validation images                   **4,127**
  Image size                      **224 × 224**

The notebook's exploratory count reports approximately **20.6K images**
across the 15 classes.

### Classes

-   Pepper --- Bell --- Bacterial Spot
-   Pepper --- Bell --- Healthy
-   Potato --- Early Blight
-   Potato --- Late Blight
-   Potato --- Healthy
-   Tomato --- Bacterial Spot
-   Tomato --- Early Blight
-   Tomato --- Late Blight
-   Tomato --- Leaf Mold
-   Tomato --- Septoria Leaf Spot
-   Tomato --- Spider Mites / Two-spotted Spider Mite
-   Tomato --- Target Spot
-   Tomato --- Tomato Yellow Leaf Curl Virus
-   Tomato --- Tomato Mosaic Virus
-   Tomato --- Healthy

------------------------------------------------------------------------

## 🧠 Model Architecture

The project uses **MobileNetV2** as the pretrained convolutional feature
extractor.

``` text
Input Image (224 × 224 × 3)
          │
          ▼
   Data Augmentation
          │
          ▼
MobileNetV2 (ImageNet)
   Transfer Learning
          │
          ▼
Global Average Pooling
          │
          ▼
      Dropout (0.2)
          │
          ▼
   Dense Softmax Layer
          │
          ▼
    15 Class Output
```

### Training strategy

**Stage 1 --- Transfer Learning**

-   MobileNetV2 pretrained on ImageNet
-   Base model initially frozen
-   Adam optimizer
-   Learning rate: `0.001`
-   Sparse categorical cross-entropy loss
-   Class weights used to compensate for class imbalance
-   Maximum of 15 epochs
-   Early stopping based on validation loss

**Stage 2 --- Fine-Tuning**

-   MobileNetV2 backbone made trainable
-   First 100 layers kept frozen
-   Remaining layers fine-tuned
-   Learning rate reduced to `1e-5`
-   Up to 10 additional epochs
-   Best weights restored using checkpointing/early stopping

------------------------------------------------------------------------

## 📊 Model Performance

The final evaluation on the **4,127-image validation set** achieved:

  Metric                     Score
  --------------------- ----------
  **Accuracy**             **94%**
  Macro Average F1        **0.93**
  Weighted Average F1     **0.94**

The classification report shows strong performance across most classes.
The model achieved particularly strong results for classes such as
Pepper Bell Healthy, Pepper Bell Bacterial Spot, Potato Early Blight,
and Tomato Yellow Leaf Curl Virus.

```
<img width="1312" height="1189" alt="confusion-matrix" src="https://github.com/user-attachments/assets/d5d05232-7bf5-4bbd-8db0-cf185ee7298f" />

```
### Confusion Matrix

The confusion matrix shows the distribution of correct and incorrect
predictions across all 15 classes. Most predictions fall along the
diagonal, indicating strong class-wise classification performance.

------------------------------------------------------------------------

## 🧪 Example Predictions

### 🔴 Pepper Bell --- Bacterial Spot

```
<img width="1916" height="973" alt="bacterial-spot-prediction" src="https://github.com/user-attachments/assets/160c7bc4-0d32-4d20-85e4-0249f8c99d01" />

```
The application correctly predicts **Pepper Bell Bacterial Spot** with
**100% displayed confidence** for the example shown in the interface.

### 🟢 Pepper Bell --- Healthy

```
<img width="1919" height="874" alt="healthy-prediction" src="https://github.com/user-attachments/assets/5a9b6eb4-ac9e-4879-9bac-9520354c7b42" />

```
The application correctly predicts **Pepper Bell Healthy** with **100%
displayed confidence** for the example shown.

> **Note:** The confidence values shown above are the model's output for
> the particular screenshots and should not be interpreted as a
> guarantee of correctness on unseen real-world images.

------------------------------------------------------------------------

## 🌐 Gradio Web Interface

The trained model is wrapped in a simple Gradio application.

Users can upload a plant leaf image and receive the predicted class
along with supporting information.

```
<img width="1915" height="916" alt="selection-menu" src="https://github.com/user-attachments/assets/4cc6244e-05d6-41c0-a207-89a94df46f5f" />

```
### Interface workflow

``` text
Upload Leaf Image
        │
        ▼
Resize to 224 × 224
        │
        ▼
CNN / MobileNetV2 Prediction
        │
        ▼
Find Highest-Probability Class
        │
        ├──► Disease Information
        │
        └──► Recommended Countermeasures
```

The interface is implemented using:

``` python
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Textbox(label="Disease Information"),
        gr.Textbox(label="Recommended Countermeasures")
    ],
    title="Plant Disease Prediction",
    description="Upload an image of a plant leaf to detect diseases."
)
```

------------------------------------------------------------------------

## 🗂️ Project Structure

A suggested GitHub repository structure is:

``` text
plant-disease-prediction/
│
├── 📓 plant-disease-prediction.ipynb
├── 🤖 plant_disease_model.keras
├── 📋 class_names.json
├── 📄 README.md
│
└── images/
    ├── interface.png
    ├── selection-menu.png
    ├── bacterial-spot-prediction.png
    ├── healthy-prediction.png
    └── confusion-matrix.png
```

> The notebook saves the trained model as `plant_disease_model.keras`
> and the class names as `class_names.json`.

------------------------------------------------------------------------

## 🛠️ Technologies Used

  Technology               Purpose
  ------------------------ ------------------------------------
  **Python**               Core programming
  **TensorFlow / Keras**   Deep learning & model training
  **MobileNetV2**          Transfer-learning backbone
  **NumPy**                Numerical operations
  **Pandas**               Dataset exploration
  **Matplotlib**           Visualization
  **Seaborn**              Confusion matrix visualization
  **Scikit-learn**         Class weights & evaluation metrics
  **Gradio**               Interactive web interface

------------------------------------------------------------------------

## 🚀 How to Run

### 1. Load the app.py, class_names.json and keras model onto Google colab

### 2. Install dependencies

``` bash
pip install tensorflow numpy gradio pillow
```

### 3. Run !python app.py in the next cell

``` text
You'll get a local link as well as a public link to access the gradio interface!
```

### Or..

Run the notebook in conjunction with the dataset to:

1.  Load and explore the PlantVillage dataset
2.  Create training and validation datasets
3.  Calculate class weights
4.  Build the MobileNetV2 transfer-learning model
5.  Train the classifier
6.  Fine-tune the model
7.  Evaluate performance
8.  Save the trained model and class names

### 4. Run the Gradio interface

After the model files have been generated, run the interface section of
the notebook.

The application launches a local Gradio web interface where a user can
upload a leaf image for prediction.

------------------------------------------------------------------------

## 📈 What This Project Demonstrates

This project brings together several practical machine-learning
concepts:

-   Exploratory data analysis
-   Image classification
-   Transfer learning
-   CNN-based feature extraction
-   Data augmentation
-   Handling class imbalance
-   Model checkpointing
-   Early stopping
-   Fine-tuning pretrained networks
-   Classification metrics
-   Confusion-matrix analysis
-   Model serialization
-   Building an ML-powered user interface

------------------------------------------------------------------------

## 🔮 Future Improvements

Possible next steps include:

-   Add more plant species and disease categories
-   Test the model on field photographs rather than primarily controlled
    dataset images
-   Add top-3/top-5 predictions
-   Display class probabilities for all classes
-   Add Grad-CAM visualizations to explain which leaf regions influenced
    the prediction
-   Improve robustness to different lighting, backgrounds, camera
    quality, and leaf orientations
-   Deploy the Gradio application as a persistent cloud service
-   Add an image-quality check to detect non-leaf or unsuitable images
-   Track user feedback to identify difficult classes and improve the
    model

------------------------------------------------------------------------

## 👨‍💻 Project

**Plant Disease Prediction using CNN & Transfer Learning**

Built with **TensorFlow/Keras + MobileNetV2 + Gradio**.
