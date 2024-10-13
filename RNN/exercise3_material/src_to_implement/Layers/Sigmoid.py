import numpy as np

class Sigmoid:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        self.sigm = 1/(1 + np.exp(-input_tensor))
        return self.sigm

    def backward(self, error_tensor):
        sigm_grad = self.sigm*(1-self.sigm)
        return sigm_grad*error_tensor
