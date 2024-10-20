import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

'''

This is a completely separae script used to testing purpose of the model using cli. To check if the given image belongs to which class

'''


# Function to classify the image based on the trained model
def classify_image(img_path):
    # Load the pre-trained model
    model = tf.keras.models.load_model('issue_classifier.h5')

    # Load the image, resize it to 224x224 (as expected by the model), and convert to array
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    # Expand dimensions to match the model's input format and normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img_array)

    # Define class names that the model was trained on
    class_names = ['security', 'electricity', 'water']

    # Get the index of the highest prediction score and map it to the class name
    predicted_class = class_names[np.argmax(predictions)]

    # Return the result based on the predicted class
    if predicted_class == 'water':
        return "Water-related issue detected"
    elif predicted_class == 'electricity':
        return "Electricity-related issue detected"
    elif predicted_class == 'security':
        return "Security issue detected"
    else:
        return "Issue not recognized"

# Main block to test the classification on different images
if __name__ == "__main__":
    # Define the paths to the test images on your Windows system
    test_image_paths = [
        'C:\\Users\\Anamay\\Downloads\\training_test\\image.png',
        'C:\\Users\\Anamay\\Downloads\\training_test\\image1.png',
        'C:\\Users\\Anamay\\Downloads\\training_test\\image2.png'
    ]

    # Loop through each image and classify it
    for img_path in test_image_paths:
        result = classify_image(img_path)
        print(f'Results--Image: {img_path}, Prediction: {result}')
