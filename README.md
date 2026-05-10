# Artificial Neural Network (ANN) - MNIST Dataset Classification

A from-scratch implementation of an Artificial Neural Network using only **NumPy** and **Matplotlib** to classify handwritten digits from the MNIST dataset.

## 📋 Project Overview

This project demonstrates how to build and train a multi-layer neural network without relying on deep learning frameworks. The implementation includes:

- **Forward Propagation**: Pass input through network layers
- **Backward Propagation**: Calculate gradients and update weights
- **Activation Functions**: ReLU, Sigmoid, Softmax
- **Loss Functions**: Cross-entropy loss
- **Optimization**: Stochastic Gradient Descent (SGD)

## 📊 MNIST Dataset

The **MNIST (Modified National Institute of Standards and Technology)** database contains:
- **60,000** training images
- **10,000** test images
- **28×28 pixel** grayscale images
- **10 classes** (digits 0-9)
- **Normalized pixel values** (0-1)

## 🛠️ Dependencies

```
numpy>=1.19.0
matplotlib>=3.3.0
```

## 📦 Installation

### 1. Clone/Download the Project
```bash
cd Dl-Practicle/Ann-for-MINIST-Dataset
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install numpy matplotlib
```

## 🚀 Usage

### Basic Example
```python
from ann_mnist import ANN
import numpy as np

# Initialize network: input_layer=784, hidden_layers=[128, 64], output_layer=10
model = ANN(layer_sizes=[784, 128, 64, 10])

# Load and prepare data
X_train, y_train = load_mnist_train()
X_test, y_test = load_mnist_test()

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Train the network
model.train(X_train, y_train, epochs=100, learning_rate=0.01, batch_size=32)

# Evaluate on test set
accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions
predictions = model.predict(X_test[:10])
```

## 📁 Project Structure

```
ann-mnist/
│
├── README.md                 # This file
├── main.py                   # Main training script
├── ann_mnist.py              # Neural network implementation
├── data_loader.py            # MNIST data loading utilities
├── visualization.py          # Plotting utilities
└── models/                   # Saved model weights (optional)
```

## 🧠 Neural Network Architecture

### Default Configuration
```
Input Layer (784 neurons)
    ↓
Hidden Layer 1 (128 neurons) - ReLU activation
    ↓
Hidden Layer 2 (64 neurons) - ReLU activation
    ↓
Output Layer (10 neurons) - Softmax activation
```

### Key Components

**Activation Functions:**
- **ReLU**: `max(0, x)` - Used in hidden layers
- **Softmax**: Probability distribution - Used in output layer
- **Sigmoid**: `1 / (1 + e^-x)` - Optional for binary problems

**Loss Function:**
- **Cross-Entropy Loss**: Measures difference between predicted and actual distribution

**Optimizer:**
- **SGD with optional Momentum**: Updates weights based on gradients

## 📈 Training Process

1. **Forward Pass**: Input → Hidden Layers → Output
2. **Compute Loss**: Compare predictions with actual labels
3. **Backward Pass**: Calculate gradients from output to input
4. **Weight Update**: Adjust weights using learning rate and gradients
5. **Repeat**: For multiple epochs until convergence

## 💾 Hyperparameters

```python
# Architecture
layer_sizes = [784, 128, 64, 10]  # Network structure

# Training
epochs = 100                        # Number of training iterations
learning_rate = 0.01              # Step size for weight updates
batch_size = 32                   # Samples per update
momentum = 0.9                    # Optional momentum factor (0-1)

# Regularization
l2_penalty = 0.0001               # L2 regularization (prevents overfitting)
```

## 📊 Expected Results

| Metric | Expected Value |
|--------|---|
| Training Accuracy | ~98% |
| Test Accuracy | ~97% |
| Training Time (100 epochs) | ~5-10 minutes |

## 🔧 Implementation Details

### Weight Initialization
- **Xavier/Glorot Initialization**: Ensures stable training
- Formula: `W ~ U[-√(6/(n_in + n_out)), √(6/(n_in + n_out))]`

### Forward Propagation
```python
z = np.dot(input, weights) + bias
a = activation(z)
```

### Backward Propagation (Chain Rule)
```python
dL/dw = (dL/dz) * (dz/dw)
```

## 📉 Visualization

The project includes visualization utilities:

```python
# Plot training loss and accuracy
plot_training_history(loss_history, accuracy_history)

# Visualize predictions
plot_predictions(images, predictions, labels)

# Display confusion matrix
plot_confusion_matrix(y_true, y_pred)
```

## ⚠️ Limitations & Considerations

- **No GPU Support**: Pure NumPy implementation (slow on large datasets)
- **Memory Usage**: Stores all layer outputs during training
- **Learning Rate**: Requires manual tuning for optimal performance
- **Convergence**: May need regularization for better generalization
- **Training Time**: Significantly slower than optimized frameworks

## 🎯 Learning Outcomes

This project helps understand:
- ✓ How neural networks work from first principles
- ✓ Matrix operations in machine learning
- ✓ Gradient-based optimization
- ✓ Backpropagation algorithm
- ✓ Classification on real-world datasets

## 📚 Resources

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- Neural Networks Basics: https://en.wikipedia.org/wiki/Artificial_neural_network
- Backpropagation Explained: https://www.youtube.com/watch?v=tIeHLnjs5U8

## 📝 License

This project is for educational purposes.

## 👤 Author

Created as part of MCA Deep Learning coursework.

---

**Happy Learning!** 🚀

