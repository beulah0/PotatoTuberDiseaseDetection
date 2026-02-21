🥔 Potato Tuber Disease Detection using Deep Learning
📌 Project Overview

This project is an AI-powered potato tuber disease detection system that automatically identifies diseases from potato images using deep learning and computer vision techniques.

The system can classify potato tubers into 5 categories:

Healthy
Blackspot Bruising
Soft Rot
Brown Rot
Dry Rot

It also provides confidence scores, top-3 predictions, and Grad-CAM visual explanations to show which image regions influenced the model’s decision.

🚀 Key Features

Deep Learning-based image classification
Transfer learning using MobileNetV2
Real-time disease prediction via Streamlit web app
Grad-CAM heatmap visualization
Data augmentation for improved accuracy
Performance metrics: Accuracy, Precision, Recall, F1-Score
Confusion matrix visualization

🧠 Technologies Used

Python
TensorFlow / Keras
OpenCV
NumPy & Pandas
Matplotlib & Seaborn
Streamlit
Scikit-learn


The system supports two approaches:

1️⃣ CNN from Scratch

Custom convolutional neural network with:

Multiple Conv layers

Batch Normalization

Dropout for regularization

2️⃣ Transfer Learning (Used in Final Model)

Base model: MobileNetV2

Frozen pretrained layers

Custom classification head

📊 Model Performance

Test Results:

Accuracy: 95.4%

Precision: 95.6%

Recall: 95.1%

F1-Score: 95.3%

🖥️ How to Run the Project
Step 1 — Clone Repository
git clone https://github.com/yourusername/PotatoDiseaseDetection.git
cd PotatoDiseaseDetection

Step 2 — Create Virtual Environment
python -m venv myenv
myenv\Scripts\activate

Step 3 — Install Dependencies
pip install -r requirements.txt

Step 4 — Train Model (Optional)
python train.py

Step 5 — Run Web App
streamlit run app.py

📸 Output Screens

The system provides:
Disease prediction

Confidence score

Top-3 predictions

Grad-CAM explanation heatmap

Disease information (symptoms, prevention, treatment)
