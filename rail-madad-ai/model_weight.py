'''
This script loads a pre-trained deep learning model and inspects its internal weights.

1. The model is loaded from the 'issue_classifier.h5' file using the `load_model()` function from TensorFlow/Keras.
2. After loading, the script retrieves all the model's weights using the `get_weights()` function.
3. It specifically inspects the weights of the first layer and prints their shape, which helps to understand the dimensions of the weight matrix for that layer.
4. Optionally, the script visualizes the distribution of the first layer's weights using a histogram.
   The weights are flattened into a 1D array using `ravel()` and then plotted with 50 bins using Matplotlib for easy visualization of the weight distribution.
'''



from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Load the model
model = load_model('issue_classifier.h5')

# Get model weights
weights = model.get_weights()

# Inspect the first layer's weights
print(weights[0].shape)  # e.g., shape of the first layer's weights

# Optionally visualize the weights
import matplotlib.pyplot as plt
plt.hist(weights[0].ravel(), bins=50)
plt.show()
