import numpy as np

class TanH:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        self.tanh = np.tanh(input_tensor)
        return self.tanh

    def backward(self, error_tensor):
        tanh_grad = 1 - self.tanh**2
        return tanh_grad*error_tensor
