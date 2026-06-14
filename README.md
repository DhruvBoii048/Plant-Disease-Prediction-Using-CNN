# 🌱 Plant Disease Prediction using Deep Learning

A Convolutional Neural Network (CNN) based plant disease classification system capable of identifying diseases across multiple crop species from leaf images.

This project was developed using TensorFlow and Keras and trained on a multi-class plant disease image dataset containing healthy and diseased plant leaves.

---

## 📌 Project Overview

Plant diseases significantly impact agricultural productivity and crop quality. Early detection helps farmers take corrective measures and reduce losses.

This project uses Deep Learning techniques to classify plant leaf images into various disease categories and healthy classes.

The model learns visual patterns from leaf images and predicts the most probable disease category for a given input image.

---

## ✨ Features

- Multi-class plant disease classification
- Deep CNN architecture built using TensorFlow/Keras
- Supports 38 plant disease categories
- Training and validation performance visualization
- Confusion matrix generation
- Detailed classification report
- Single-image disease prediction script
- Saved model for inference and deployment

---

## 📂 Project Structure

```text
Plant-Disease-Prediction/
│
├── train_plant_disease.py        # Model training pipeline
├── test/
│   ├── test/
│   │   ├── sample_images...
│   │
│   └── test_plant_disease.py     # Single image prediction
│
├── training_hist.json            # Training history
├── README.md
├── .gitignore
│
└── Dataset/
    ├── train/
    ├── valid/
```

---

## 🧠 Model Architecture

The CNN architecture consists of:

- 5 Convolutional Blocks
- ReLU Activation Functions
- Max Pooling Layers
- Dropout Regularization
- Fully Connected Dense Layer
- Softmax Output Layer

### Layer Configuration

| Layer Type | Filters |
|------------|----------|
| Conv2D | 32 |
| Conv2D | 32 |
| Conv2D | 64 |
| Conv2D | 64 |
| Conv2D | 128 |
| Conv2D | 128 |
| Conv2D | 256 |
| Conv2D | 256 |
| Conv2D | 512 |
| Conv2D | 512 |

Additional layers:

- Dropout (0.25)
- Dense (1500 neurons)
- Dropout (0.40)
- Dense (38 output classes)

---

## ⚙️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- OpenCV
- Scikit-Learn

---

## 📊 Dataset

The model was trained on a publicly available Plant Disease dataset containing images of healthy and diseased plant leaves.

### Dataset Characteristics

- 38 Classes
- RGB Images
- Image Size: 128 × 128
- Multiple crop species
- Healthy and diseased samples

Example crops include:

- Apple
- Tomato
- Potato
- Corn
- Grape
- Peach
- Strawberry
- Orange
- Pepper
- Soybean

---

## 🚀 Training

Run the training pipeline:

```bash
python train_plant_disease.py
```

The script performs:

1. Dataset loading
2. Image preprocessing
3. CNN model creation
4. Model training
5. Validation
6. Performance evaluation
7. Model saving

Generated outputs:

- trained_model.keras
- training_hist.json

---

## 🔍 Testing on a Single Image

The project includes a separate inference script.

Update the image path:

```python
image_path = "test/test/Pototato_Healthy8.JPG"
```

Run:

```bash
python test/test_plant_disease.py
```

The script:

- Loads the trained model
- Reads the test image
- Performs prediction
- Displays the predicted disease class

---

## 📈 Evaluation Metrics

The training pipeline generates:

- Accuracy
- Validation Accuracy
- Classification Report
- Confusion Matrix

Metrics are computed using:

```python
classification_report()
confusion_matrix()
```

from Scikit-Learn.

---

## 🖼️ Example Prediction Workflow

Input:

```text
Leaf Image
```

↓

CNN Feature Extraction

↓

Disease Classification

↓

Output:

```text
Tomato___Late_blight
```

---

## 💾 Model Saving

The trained model is saved using:

```python
model.save("trained_model.keras")
```

This allows future inference without retraining.

---

## 🔮 Future Improvements

- Transfer Learning using EfficientNet or ResNet
- Mobile Deployment
- Real-time Camera Prediction
- Web Application Integration
- Disease Severity Estimation
- Treatment Recommendation System

---

## 👨‍💻 Author

**Dhruv Maheshwari**

Deep Learning • Computer Vision • Machine Learning

---

## 📜 License

This project is intended for educational and research purposes.