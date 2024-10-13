# Fully Connected Neural Network (FCN) from Scratch

This project implements a Fully Connected Neural Network (FCN) entirely from scratch in Python, using basic numpy operations. The implementation covers key components like layers, activation functions, loss functions, optimizers, and the forward and backward passes necessary for training the network.

## Features
- Implementation of a Fully Connected Neural Network (FCN) without any high-level libraries like TensorFlow or PyTorch.
- Custom layers, including FullyConnected, ReLU, and SoftMax layers.
- Cross-Entropy Loss for classification tasks.
- SGD (Stochastic Gradient Descent) as an optimizer.
- Forward and backward propagation from scratch.

## Project Structure

```bash
.
├── Layers
│   ├── Base.py             # Base Layer Class
│   ├── FullyConnected.py    # Fully Connected Layer
│   ├── ReLU.py              # ReLU Activation Layer
│   ├── SoftMax.py           # Softmax Layer
├── Loss.py                  # Cross-Entropy Loss Function
├── NeuralNetwork.py         # Main Neural Network Class
├── Optimizers.py            # SGD Optimizer Class
└── README.md                # Project Documentation
```
---
## Code Explanation
### Layers
**Base Layer (Base.py):**

-The base class for all layers, defines a trainable property.

**Fully Connected Layer (FullyConnected.py):**
- Implements a fully connected layer with forward and backward propagation.
- Uses bias for each input.
- Supports training and optimization of weights.
  
**ReLU Layer (ReLU.py):**
- Implements ReLU (Rectified Linear Unit) activation function.
- Provides both forward and backward propagation functions.

**Softmax Layer (SoftMax.py):**
- Softmax activation for multi-class classification.
- Converts logits into probabilities.

### Loss

**Cross-Entropy Loss (Loss.py):**
- Implements forward and backward passes for calculating the cross-entropy loss, typically used in classification tasks.

### Optimizers

**SGD Optimizer (Optimizers.py):**
- Stochastic Gradient Descent (SGD) with a simple learning rate update rule for weight optimization.

### Neural Network
**Neural Network (NeuralNetwork.py):**
- Handles the structure and workflow of the neural network.
- Supports adding layers, forward pass, backpropagation, and updating weights through training.

---
## How to Run
- 1. Clone the repository:
```bash
git clone https://github.com/your_username/fcn-from-scratch.git
```

- 2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

- 3. To train the neural network:

```python
from NeuralNetwork import NeuralNetwork
from Optimizers import Sgd
from Layers.FullyConnected import FullyConnected
from Layers.ReLU import ReLU
from Layers.SoftMax import SoftMax
from Loss import CrossEntropyLoss
# Define your data layer and loss layer
# Initialize Neural Network
```

- 4. Implement Training Loop according to your requiremnets.
---
## Contributors
- Email: vipulpatil7057@gmailcom
- Email: sushantnemade15@gmail.com
