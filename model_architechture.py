'''
This script is designed to load and analyze a pre-trained deep learning model saved in the 'issue_classifier.h5' file.

1. The script uses TensorFlow/Keras to load the trained model using `load_model()`.
2. Once the model is loaded, it prints a summary of the model's architecture, including layers, input/output shapes, and number of parameters, to help understand the structure.
3. Additionally, the script offers the option to generate a visual representation of the model architecture and save it as a PDF file ('model_architecture.pdf').
   This plot includes the shape of each layer, making it useful for better visual understanding of the model's structure.
'''



from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Load the model
model = load_model('issue_classifier.h5')

# Print model summary to understand architecture
model.summary()

# Optionally, visualize the model structure as a plot
plot_model(model, to_file='model_architecture.pdf', show_shapes=True)


