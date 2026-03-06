# 🦠 Malaria Detection using Deep Learning

## 🚀 Project Overview

This project implements an AI-powered malaria detection system using a Custom Convolutional Neural Network (CNN) trained on microscopic blood smear images.

The model classifies red blood cells into:

- **Parasitized (Infected)**
- **Uninfected (Healthy)**

A Flask-based web application allows users to upload microscopic images and receive real-time predictions with confidence scores.

This project demonstrates a complete end-to-end deep learning pipeline:
Data Collection → Preprocessing → Model Training → Evaluation → Web Deployment.

---

## 📊 Dataset

**Source:** NIH Malaria Dataset (Kaggle)  
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria  

### Dataset Details

- Total Images Used: 998
- Parasitized: 499
- Uninfected: 499
- Image Type: Thin blood smear microscopy
- Format: PNG (RGB)
- Balanced Dataset (50-50 split)
- Staining Method: Giemsa

Images were resized and normalized before training.

---

## 🧠 Model Architecture

### Custom CNN

The model includes:

- Convolutional layers for feature extraction
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Dense fully connected layers
- Sigmoid activation for binary classification

### Training Techniques Used

- Data Augmentation
- Early Stopping
- ReduceLROnPlateau
- Model Checkpoint
- Learning Rate Scheduling

---

## 📈 Model Performance

The model was evaluated using a validation dataset.

**Validation Accuracy:** ~96%  
**Precision:** ~96%  
**Recall:** ~96%  
**F1-Score:** ~96%  
**ROC-AUC:** ~0.98  

The model demonstrates strong class separability and stable generalization.

---

## 📋 Evaluation Metrics

The following metrics were used to evaluate model performance:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve

Example Classification Output:

Accuracy: 0.96  
Precision: 0.96  
Recall: 0.96  
F1-Score: 0.96  

---

## 🖥 Web Application Features

- Image upload functionality
- Image preview before analysis
- Real-time prediction
- Confidence score visualization
- Professional UI design
- Timestamp logging
- Medical disclaimer
- Dynamic result rendering using AJAX + JSON

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Flask
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- HTML / CSS / JavaScript

---

## 📂 Project Structure

```
MALARIA_DETECTION/
│
├── app.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
│
├── models/
│     └── malaria_model.h5
│
├── templates/
│     └── index.html
│
├── static/
│     └── uploads/
│
├── screenshots/
│
└── dataset/   (Not included in repository)
```

---

## ⚙ How to Run the Project

### 1️⃣ Clone the Repository

```
git clone <your_repo_url>
cd MALARIA_DETECTION
```

### 2️⃣ Create Virtual Environment (Recommended)

```
python -m venv venv
venv\Scripts\activate   (Windows)
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 🔮 Future Improvements

- Grad-CAM for explainable AI
- Multi-species malaria classification
- Mobile deployment using TensorFlow Lite
- Cloud deployment (AWS / Render)
- REST API integration

---

## ⚠ Medical Disclaimer

This AI system is intended as a screening tool only.  
Results must be verified by qualified healthcare professionals.  
It is not a substitute for professional medical diagnosis.

---

## 👨‍💻 Author

This project demonstrates:

- Deep learning model design
- Model evaluation and performance analysis
- Flask-based deployment
- End-to-end AI workflow integration

---

⭐ If you found this project useful, consider giving it a star.