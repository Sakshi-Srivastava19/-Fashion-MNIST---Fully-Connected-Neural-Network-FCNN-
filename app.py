import streamlit as st
import numpy as np
#import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist

# Load dataset
(x_train, y_train), (_, _) = fashion_mnist.load_data()
x_train = x_train / 255.0  # Normalize
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the trained model
model = load_model('fashion_mnist_fcnn_model.h5')

# Streamlit App UI
st.title("👗 Fashion MNIST Classifier")
st.write("Using a Fully Connected Neural Network (fcnn) to classify fashion images.")

# Select image index
index = st.slider("Select Image Index (0 - 59999)", min_value=0, max_value=59999, value=0)

# Display selected image
st.image(x_train[index], caption=f"Actual: {class_names[y_train[index]]}", width=200, channels="GRAY")

# Predict
pred = model.predict(x_train[index].reshape(1, 28, 28))
predicted_class = np.argmax(pred)
confidence = np.max(pred) * 100

# Display prediction
st.markdown(f"### 🔍 Prediction: **{class_names[predicted_class]}**")
st.markdown(f"**Confidence:** {confidence:.2f}%")

# Display confidence chart
st.bar_chart(pred.flatten())
