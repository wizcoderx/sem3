'''
This script is called directly in app.py via the /image route.
The classify_image() function is responsible for downloading the image, processing it, and classifying it using the trained model (issue_classifier.h5).

'''


import os
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

model_path = os.path.join(os.path.dirname(__file__), 'issue_classifier.h5')
model = tf.keras.models.load_model(model_path)
class_names = ['electricity', 'security', 'water']

def download_image(image_url):
    """Download an image from a URL and return a PIL Image object."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
        return None

def classify_image(image_url):
    """Classify an image from a URL using the trained model."""
    img = download_image(image_url)
    if img is None:
        return "Failed to download image or image is not valid."
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    if predicted_class == 'water':
        return "Water-related issue detected"
    elif predicted_class == 'electricity':
        return "Electricity-related issue detected"
    elif predicted_class == 'security':
        return "Security issue detected"
    else:
        return "Issue not recognized"



