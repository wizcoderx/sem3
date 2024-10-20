"""
This script contains another version of the classify_image() function, similar to the one in train_model.py, but it is not directly called from app.py. This script is used for testing the image classification locally.

It is useful for manually testing image classification outside of the Flask application. While it shares logic with train_model.py, it doesn't directly influence app.py.

"""


import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def classify_image(img_path):
    model = tf.keras.models.load_model('issue_classifier.h5')

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    class_names = ['security', 'electricity', 'water']
    predicted_class = class_names[np.argmax(predictions)]

    if predicted_class == 'water':
        return "Water-related issue detected"
    elif predicted_class == 'electricity':
        return "Electricity-related issue detected"
    elif predicted_class == 'security':
        return "Security issue detected"
    else:
        return "Issue not recognized"

if __name__ == "__main__":
    test_image_paths = [
        'C:\\Users\\Anamay\\Downloads\\training_test\\image.png',
        'C:\\Users\\Anamay\\Downloads\\training_test\\image1.png',
        'C:\\Users\\Anamay\\Downloads\\training_test\\image2.png'
    ]

    for img_path in test_image_paths:
        result = classify_image(img_path)
        print(f'\n Results--Image: {img_path}, Prediction: {result}')


