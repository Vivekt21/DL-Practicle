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

# Deep Learning Lab - Artificial Neural Network from Scratch

## Project Overview

This project implements a **Fully Connected Artificial Neural Network (ANN)** from scratch using only **NumPy** and **Matplotlib** (no deep learning libraries like TensorFlow or PyTorch). The goal is to understand the fundamental concepts of neural networks, backpropagation, and optimization by building and experimenting with a custom implementation.

## Project Scope

### Datasets
- **Pseudo Image 1**: 20x30 pixel images (600 features)
- **Pseudo Image 2**: 30x60 pixel images (1800 features)

### Experiments
The project involves systematic experimentation with:

1. **Activation Functions**
   - ReLU (Rectified Linear Unit)
   - Sigmoid
   - Tanh

2. **Learning Rates**
   - 0.001 (low)
   - 0.01 (medium)
   - 0.1 (high)
   - Custom values for fine-tuning

3. **Epoch Sizes**
   - Small: 10-50 epochs
   - Medium: 100-200 epochs
   - Large: 500+ epochs

## Dependencies

```
numpy          # Numerical computations
matplotlib     # Data visualization
python >= 3.7
```

### Installation

```bash
pip install numpy matplotlib
```

## Project Structure

```
DL LAB/
├── README.md                 # Project documentation
├── Mid Term/
│   ├── Midterm1.ipynb       # Experiment set 1 - ANN implementation and basic experiments
│   └── Midterm2.ipynb       # Experiment set 2 - Advanced experiments and comparisons
└── [utils/ or notebooks/]   # Additional helper files (if needed)
```

## Key Components

### 1. Neural Network Architecture

The custom ANN includes:

- **Forward Propagation**: Computing network output given input
- **Activation Functions**: Implementing ReLU, Sigmoid, Tanh, and Linear
- **Backward Propagation**: Computing gradients for all weights and biases
- **Gradient Descent**: Updating parameters to minimize loss
- **Loss Function**: Mean Squared Error (MSE) or Cross-Entropy

### 2. Hyperparameter Tuning

Each experiment combines:
- Different network architectures
- Various activation functions
- Multiple learning rates
- Different training durations (epochs)

### 3. Evaluation Metrics

- **Training Loss**: Loss over training iterations
- **Validation/Test Accuracy**: Model performance on unseen data
- **Convergence Speed**: How quickly the model learns
- **Final Performance**: Accuracy achieved after training

## Experiment Workflow

### Midterm 1: Foundation & Basic Experiments
- Implement ANN from scratch
- Build forward and backward propagation
- Test with simple activation functions (ReLU, Sigmoid)
- Establish baseline performance
- Visualize training curves

### Midterm 2: Advanced Experiments & Analysis
- Compare all activation functions systematically
- Test various learning rates
- Experiment with different epoch sizes
- Analyze impact of hyperparameters
- Generate comparative visualizations
- Document findings and observations

## How to Run

1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Navigate to Mid Term folder** and open either `Midterm1.ipynb` or `Midterm2.ipynb`

3. **Run cells sequentially** to:
   - Build the neural network
   - Train on pseudo images
   - Visualize results
   - Compare experiments

## Expected Outputs

For each experiment, you should generate:

- **Training Curves**: Loss vs. epochs
- **Activation Function Comparison**: Performance metrics across different activations
- **Learning Rate Impact**: How different learning rates affect convergence
- **Epoch Analysis**: Training time vs. accuracy trade-offs
- **Visualizations**: Loss curves, accuracy plots, heatmaps of learned features

## Key Insights to Explore

1. **Activation Functions**: Which works best for your pseudo images?
2. **Learning Rate Sensitivity**: How does learning rate affect convergence speed?
3. **Overfitting**: Does increased epochs lead to overfitting?
4. **Trade-offs**: Balance between accuracy and training time
5. **Network Capacity**: How does network size impact learning?

## Useful Visualization Techniques

Using Matplotlib:
- Plot training/validation loss curves
- Create bar charts comparing activation functions
- Generate heatmaps of learned weights
- Visualize pseudo image samples
- Compare convergence speeds

## References

- **Backpropagation**: The chain rule for computing gradients
- **Gradient Descent**: Optimization algorithm for updating weights
- **Activation Functions**: Non-linear transformations for neural networks
- **Loss Functions**: Metrics to minimize during training

## 📝 License

This project is for educational purposes.

## 👤 Author

Created as part of MCA Deep Learning coursework.

---

**Happy Learning!** 🚀

