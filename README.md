
# Issue Classifier - Deep Learning Model

This project uses a deep learning model to classify issues based on input images. It has been trained to identify different types of issues like **security issues**, **water-related issues**, etc. The model architecture and its layers are based on TensorFlow/Keras, and the project includes scripts for loading the model, visualizing its architecture, and analyzing its weights.

## Features
- Classify images into categories (e.g., security issue, water-related issue, etc.).
- Visualization of the model architecture and weights.
- Histogram analysis of the model's first layer weights.

## Prerequisites
- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib (for visualizations)

You can install the dependencies by running the following command:
```bash
pip install tensorflow matplotlib
```

## Files in the Repository

1. **app.py**: Main application script for loading and running the model on test data.
2. **model_architecture.pdf**: Visual representation of the deep learning model's architecture.
3. **issue_classifier.h5**: Pre-trained deep learning model used for classifying images.
4. **weights_visualization.py**: Script for visualizing and analyzing the weights of the first layer in the model.
5. **README.md**: This file providing an overview of the project.

## Usage

### Loading the Model and Viewing Architecture
The model can be loaded and its summary viewed by running the following script:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Load the model
model = load_model('issue_classifier.h5')

# Print model summary
model.summary()

# Visualize the model architecture
plot_model(model, to_file='model_architecture.pdf', show_shapes=True)
```

### Visualizing Model Weights
To visualize the weights of the first layer in the model, run the following script:

```python
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the model
model = load_model('issue_classifier.h5')

# Get weights of the first layer
weights = model.get_weights()

# Visualize the weights as a histogram
plt.hist(weights[0].ravel(), bins=50)
plt.show()
```

### Output
The model will output one of the predefined categories based on the image input. For example, it might output `security issue` if it detects a security-related situation in the input image.

## Understanding the Weight Visualization
The histogram generated from the weights of the first layer helps to understand how the model's neurons are initialized and how it learns from the dataset. A large concentration of weights near zero suggests strong regularization or less significant features.

## Contributing
Feel free to contribute by submitting pull requests. Any contributions that improve the model, add new features, or enhance documentation are welcome!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
