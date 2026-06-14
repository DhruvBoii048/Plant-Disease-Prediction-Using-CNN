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

## Project Structure

```text
Plant-Disease-Prediction/
│
├── Dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── test_plant_disease.py
├── train_plant_disease.py
├── training_hist.json
├── requirements.txt
├── .gitignore
├── README.md
│
└── trained_model.h5   (generated after training)
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

## Dataset

This project uses the **New Plant Diseases Dataset (Augmented)** containing approximately **87,000 RGB images** of healthy and diseased crop leaves across **38 disease categories**. The dataset is already organized into training and validation folders and includes a separate test directory for inference.

**Dataset Source:**

https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?select=New+Plant+Diseases+Dataset%28Augmented%29

### Dataset Structure

```text
Dataset/
├── train/
├── valid/
└── test/
```

The dataset covers multiple crops including:

* Apple
* Blueberry
* Cherry
* Corn
* Grape
* Orange
* Peach
* Pepper
* Potato
* Raspberry
* Soybean
* Squash
* Strawberry
* Tomato

with a total of **38 disease and healthy plant classes**.

---

## Training the Model

After downloading and extracting the dataset into the `Dataset/` directory, train the CNN model using:

```bash
python train_plant_disease.py
```

The script will:

* Load training and validation images
* Train the CNN model
* Evaluate performance
* Save the trained model as:

```text
trained_model.h5
```
*Note:* Other saving formats can be used, and accordingly change the test_plant_disease.py
* Save training history to:

```text
training_hist.json
```

---

## 🔍 Testing on a Single Image

The project includes a separate inference script.

Update the image path:

```python
image_path = "test/test/Pototato_Healthy8.JPG"
```

Run:

```bash
python test_plant_disease.py
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
model.save("trained_model.h5")
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