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
    - ```forward(input_tensor): Computes the output of the layer.```
    - ```backward(error_tensor): Computes the gradient and passes it back to earlier layers.```
  
- **ReLU Layer (Relu.py)**
  - Applies the ReLU activation function element-wise, outputting max(0, x) for each input.

  - Key Methods:
    - ```forward(input_tensor): Computes ReLU on the input.```
    - ```backward(error_tensor): Passes the gradient for backpropagation.```

- **SoftMax Layer (SoftMax.py)**
  - Implements the SoftMax activation function, which normalizes the output into probability distribution.

  - Key Methods:
    - ```forward(input_tensor): Computes the SoftMax probabilities.```
    - ```backward(error_tensor): Calculates the gradients for backpropagation.```
      
- **Convolutional Layer (Conv.py)**
  - Implements the convolution operation. This layer applies kernels (filters) over the input to extract spatial features. Supports multiple kernels, padding, and strides.

  - Key Methods:
    - ```forward(input_tensor): Convolves the input with learnable filters (weights) and adds bias.```
    - ```backward(error_tensor): Computes gradients with respect to weights and propagates errors to previous layers.```

- **Pooling Layer (Pooling.py)**
  - Implements max pooling, which downsamples the input by taking the maximum value in each pooling region. The pooling shape and stride can be customized.

  - Key Methods:
    - ```forward(input_tensor): Applies max pooling to the input.```
    - ```backward(error_tensor): Propagates the error for backpropagation based on the pooled regions.```

- **Flatten Layer (Flatten.py)**
  - Flattens the input from a multi-dimensional tensor (e.g., for images) into a 1D tensor, which is often required before passing data to a fully connected layer.

  - Key Methods:
    - ```forward(input_tensor): Flattens the input.```
    - ```backward(error_tensor): Reshapes the error tensor for backpropagation.```
---
    
### Initialization
- **Initializers (initializers.py)**
  - Defines several initialization methods to initialize weights and biases of layers:
    - **Constant:** Initializes the weights to a constant value.
    - **UniformRandom:** Initializes weights randomly from a uniform distribution.
    - **Xavier:** An initialization scheme suited for layers with sigmoid or tanh activations.
    - **He:** An initialization scheme suited for layers with ReLU activations.
---

### Loss

- **Cross-Entropy Loss (Loss.py):**
  - Implements forward and backward passes for calculating the cross-entropy loss, typically used in classification tasks.
  - Key Methods:
    - ```forward(predicted_tensor, label_tensor): Computes the loss.```
    - ```backward(label_tensor): Computes the gradient with respect to the predictions.```
---
### Optimizers

- **SGD Optimizer (Optimizers.py):**
  - Implements optimization techniques used during training to update the weights and biases:
    - **SGD:** Stochastic Gradient Descent without momentum.
    - **SGD with Momentum:** Adds momentum to standard SGD for faster convergence.
---

### Neural Network
- **Neural Network (NeuralNetwork.py):**
  - Handles the structure and workflow of the neural network.
  - Supports adding layers, forward pass, backpropagation, and updating weights through training.

---
## How to Run

1. Clone the repository:
```bash
git clone https://github.com/vip7057/Deep-Neural-Networks-from-Scratch.git
cd Deep-Neural-Networks-from-Scratch/CNN
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. To build a CNN using these components:
Create a network with layers like Conv, Pooling, Flatten, and FullyConnected, and train it using optimizers like SGD. Here's an example to get you started:

```python
from Layers.Conv import Conv
from Layers.Pooling import Pooling
from Layers.FullyConnected import FullyConnected
from Layers.Relu import Relu
from Layers.Flatten import Flatten
from Initializers import He
from Optimizers import Adam

# Initialize the network
conv_layer = Conv(stride_shape=(1, 1), convolution_shape=(1, 3, 3), num_kernels=8)
pooling_layer = Pooling(stride_shape=(2, 2), pooling_shape=(2, 2))
flatten_layer = Flatten()
fc_layer = FullyConnected(64, 10)
relu_layer = Relu()

# Example forward pass
input_tensor = np.random.rand(5, 1, 28, 28)  # Batch size 5, 1 channel, 28x28 images
output = conv_layer.forward(input_tensor)
output = pooling_layer.forward(output)
output = flatten_layer.forward(output)
output = fc_layer.forward(output)
output = relu_layer.forward(output)ing to your requirements.
```
---
## Contributors
- Email: vipulpatil7057@gmail.com
- Email: sushantnemade15@gmail.com


