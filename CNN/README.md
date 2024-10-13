<<<<<<< HEAD:README.md
# Convolutional Neural Network (CNN) from Scratch

This project implements a Convolutional Neural Network (CNN) entirely from scratch in Python, using basic numpy operations. The core modules include essential layers for CNNs such as convolution, pooling, flattening, and fully connected layers. Additionally, several initializers, loss functions, and optimizers are implemented to provide a flexible training environment for the network.

![CNN Architecture](Illustration-of-a-fully-connected-neural-network.png)

## Features
- Implementation of a Convolutional Neural Network (CNN) without any high-level libraries like TensorFlow or PyTorch.


## Project Structure

```bash
.
├── Layers
│   ├── Base.py              # Base Layer Class
│   ├── Conv.py              # Convolution Layer Class
│   ├── Flatten.py           # Flatten Layer Class
│   ├── FullyConnected.py    # Fully Connected Layer
│   ├── Pooling.py           # Pooling Layer Class
│   ├── ReLU.py              # ReLU Activation Layer
│   ├── SoftMax.py           # Softmax Layer
├── Initializers.py          # Different weight and bias initializers
├── Loss.py                  # Cross-Entropy Loss Function
├── NeuralNetwork.py         # Main Neural Network Class
├── Optimizers.py            # SGD and SGD with Momentum Optimizer Class
└── README.md                # Project Documentation
```
---
## Code Explanation
### Layers
- **Base Layer (Base.py)**
  - Provides the basic structure and interface for all layers. It handles common attributes like trainable and optimizer, and defines the essential forward() and backward() methods which are inherited by other layers.

- **Fully Connected Layer (FullyConnected.py)**
  - A fully connected (dense) layer that connects each input node to every output node. The weights and biases are trainable and can be updated using optimizers during training.

- Key Methods:
    -[forward(input_tensor)]: Computes the output of the layer.
    -[backward(error_tensor)]: Computes the gradient and passes it back to earlier layers.
  
3. ReLU Layer (Relu.py)
Applies the ReLU activation function element-wise, outputting max(0, x) for each input.

Key Methods:

forward(input_tensor): Computes ReLU on the input.
backward(error_tensor): Passes the gradient for backpropagation.
4. SoftMax Layer (SoftMax.py)
Implements the SoftMax activation function, which normalizes the output into probability distribution.

Key Methods:

forward(input_tensor): Computes the SoftMax probabilities.
backward(error_tensor): Calculates the gradients for backpropagation.
5. Convolutional Layer (Conv.py)
Implements the convolution operation. This layer applies kernels (filters) over the input to extract spatial features. Supports multiple kernels, padding, and strides.

Key Methods:

forward(input_tensor): Convolves the input with learnable filters (weights) and adds bias.
backward(error_tensor): Computes gradients with respect to weights and propagates errors to previous layers.
initialize(weights_initializer, bias_initializer): Initializes the weights and biases using specified initialization methods.
6. Pooling Layer (Pooling.py)
Implements max pooling, which downsamples the input by taking the maximum value in each pooling region. The pooling shape and stride can be customized.

Key Methods:

forward(input_tensor): Applies max pooling to the input.
backward(error_tensor): Propagates the error for backpropagation based on the pooled regions.
7. Flatten Layer (Flatten.py)
Flattens the input from a multi-dimensional tensor (e.g., for images) into a 1D tensor, which is often required before passing data to a fully connected layer.

Key Methods:

forward(input_tensor): Flattens the input.
backward(error_tensor): Reshapes the error tensor for backpropagation.
8. Initializers (initializers.py)
Defines several initialization methods to initialize weights and biases of layers:

Constant: Initializes the weights to a constant value.
UniformRandom: Initializes weights randomly from a uniform distribution.
Xavier: An initialization scheme suited for layers with sigmoid or tanh activations.
He: An initialization scheme suited for layers with ReLU activations.
9. Optimizers (optimizers.py)
Implements optimization techniques used during training to update the weights and biases:

SGD: Stochastic Gradient Descent without momentum.
SGD with Momentum: Adds momentum to standard SGD for faster convergence.
10. Loss Functions (loss.py)
Implements the cross-entropy loss function commonly used for classification tasks.

Key Methods:

forward(predicted_tensor, label_tensor): Computes the loss.
backward(label_tensor): Computes the gradient with respect to the predictions.

### Loss

- **Cross-Entropy Loss (Loss.py):**
  - Implements forward and backward passes for calculating the cross-entropy loss, typically used in classification tasks.

### Optimizers

- **SGD Optimizer (Optimizers.py):**
  - Stochastic Gradient Descent (SGD) with a simple learning rate update rule for weight optimization.

### Neural Network
- **Neural Network (NeuralNetwork.py):**
  - Handles the structure and workflow of the neural network.
  - Supports adding layers, forward pass, backpropagation, and updating weights through training.

---
## How to Run

1. Clone the repository:
```bash
git clone https://github.com/your_username/fcn-from-scratch.git
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. To train the neural network:

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
4. Implement a Training Loop according to your requirements.
---
## Contributors
- Email: vipulpatil7057@gmail.com
- Email: sushantnemade15@gmail.com
=======
# Fully Connected Neural Network (FCN) from Scratch

This project implements a Fully Connected Neural Network (FCN) entirely from scratch in Python, using basic numpy operations. The implementation covers key components like layers, activation functions, loss functions, optimizers, and the forward and backward passes necessary for training the network.

![FCN Architecture](https://www.researchgate.net/profile/Faiq-Khalid/publication/333336147/figure/fig2/AS:767975674093575@1560111077953/Illustration-of-a-fully-connected-neural-network.png)

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
- **Base Layer (Base.py):**
  - The base class for all layers, defines a trainable property.

- **Fully Connected Layer (FullyConnected.py):**
  - Implements a fully connected layer with forward and backward propagation.
  - Uses bias for each input.
  - Supports training and optimization of weights.
  
- **ReLU Layer (ReLU.py):**
  - Implements ReLU (Rectified Linear Unit) activation function.
  - Provides both forward and backward propagation functions.

- **Softmax Layer (SoftMax.py):**
  - Softmax activation for multi-class classification.
  - Converts logits into probabilities.

### Loss

- **Cross-Entropy Loss (Loss.py):**
  - Implements forward and backward passes for calculating the cross-entropy loss, typically used in classification tasks.

### Optimizers

- **SGD Optimizer (Optimizers.py):**
  - Stochastic Gradient Descent (SGD) with a simple learning rate update rule for weight optimization.

### Neural Network
- **Neural Network (NeuralNetwork.py):**
  - Handles the structure and workflow of the neural network.
  - Supports adding layers, forward pass, backpropagation, and updating weights through training.

---
## How to Run

1. Clone the repository:
```bash
git clone https://github.com/your_username/fcn-from-scratch.git
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. To train the neural network:

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
4. Implement a Training Loop according to your requirements.
---
## Contributors
- Email: vipulpatil7057@gmail.com
- Email: sushantnemade15@gmail.com
>>>>>>> bcda3b2 (added readme):FCN/README.md
