import numpy as np
class Flatten:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        input_tensor = np.array(input_tensor)
        self.input_shape = input_tensor.shape
        return input_tensor.flatten().reshape(input_tensor.shape[0], -1)

    def backward(self, error_tensor):

        return np.reshape(error_tensor, self.input_shape)

        
