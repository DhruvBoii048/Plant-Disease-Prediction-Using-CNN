"""# Importing Libraries"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


"""# Loading Model"""

model = tf.keras.models.load_model('trained_model.h5')
model.summary()


"""# Visualizing Single Image of Test set"""

import cv2
image_path = "Dataset/test/TomatoHealthy2.JPG"
## Reading the image
img = cv2.imread(image_path) # loads image in BGR format by default
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #converting BGR image to RGB

## Dislaying image
plt.imshow(img)
plt.title("Test Image")
plt.xticks([])
plt.yticks([])
plt.show()


"""# Testing Model"""

image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
input_array = tf.keras.preprocessing.image.img_to_array(image)
input_array = np.array([input_array]) # converting single image to a batch
print(input_array.shape)

prediction = model.predict(input_array)
prediction,prediction.shape

result_index = np.argmax(prediction)
result_index

class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


"""# Display result of disease prediction"""
model_prediction = class_name[result_index]
plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()