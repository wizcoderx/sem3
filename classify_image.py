"""
This script contains another version of the classify_image() function, similar to the one in train_model.py, but it is not directly called from app.py. This script is used for testing the image classification locally.

It is useful for manually testing image classification outside of the Flask application. While it shares logic with train_model.py, it doesn't directly influence app.py.

"""

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def classify_image(img_path):
    try:
        model = tf.keras.models.load_model('issue_classifier.h5')
        logging.debug(f"Model loaded successfully from 'issue_classifier.h5'")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        predictions = model.predict(img_array)
        logging.debug(f"Predictions: {predictions}")

        class_names = ['security', 'electricity', 'water']
        predicted_class = class_names[np.argmax(predictions)]

        return f"{predicted_class.capitalize()} issue detected" if predicted_class in class_names else "Issue not recognized"
    except Exception as e:
        logging.error(f"Error processing image or making predictions: {e}")
        raise

