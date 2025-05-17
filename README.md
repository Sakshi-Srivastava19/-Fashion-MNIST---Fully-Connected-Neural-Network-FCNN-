# -Fashion-MNIST---Fully-Connected-Neural-Network-FCNN-
🧠 Fashion MNIST - Fully Connected Neural Network (FCNN)

# 🧠 Fashion MNIST - Fully Connected Neural Network (FCNN)

This project implements a Fully Connected Neural Network (FCNN) using TensorFlow and Keras to classify images from the **Fashion MNIST dataset**, which contains grayscale images of various fashion items.

---

## 📁 Dataset

The **Fashion MNIST** dataset contains:
- 60,000 training images
- 10,000 testing images  
Each image is 28x28 pixels in grayscale, and belongs to one of 10 clothing categories.

---

## 🏗️ Model Architecture

The model is a simple feedforward neural network:
- **Input Layer:** Flatten (28x28)
- **Hidden Layers:** 
  - Dense (128 neurons, ReLU)
  - Dense (64 neurons, ReLU)
- **Output Layer:** Dense (10 neurons, Softmax)

---

## 📊 Data Preprocessing

- Normalized pixel values to the range `[0, 1]`
- Labels were one-hot encoded for multi-class classification
- Visualized sample images to understand dataset distribution

---

## ⚙️ Training Configuration

- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Metrics:** Accuracy
- **Epochs:** 20
- **Validation Split:** 20%

---

## 📉 Model Evaluation

- Training and validation accuracy/loss were plotted for visual inspection.
- Final test accuracy was evaluated on the test set.

---

## 🧪 Predictions

Predictions can be made directly using the trained model:

```python
pred = model.predict(x_train[110].reshape(1, 28, 28))
print(f"Label : {pred.argmax()} , Class : {class_names[pred.argmax()]}")
```

Visualization:

```python
def see_image(image, label):
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap='gray')
    plt.title(f"Actual Label: {class_names[label]}")
    plt.axis('off')
    plt.show()

see_image(x_train[110], y_train[110])
```

---

## 💾 Saving the Model

```python
model.save('fashion_mnist_fcnn_model.h5')
```

---

## 📦 Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install tensorflow matplotlib
```

---

## 📌 Results

The FCNN achieved high accuracy on both training and validation sets, showing good generalization without overfitting.

---

## 📬 Contact

For any questions or help, reach out to:  
📧 **support@intellipaat.com**

---

© 2025 Intellipaat — All rights reserved.
