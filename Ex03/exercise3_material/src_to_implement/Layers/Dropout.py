import numpy as np
from Exercise.Ex03.exercise3_material.src_to_implement.Layers.Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):
        if self.testing_phase == False:
            self.mask = (np.random.rand(*input_tensor.shape) < self.probability).astype(int)*(1 / self.probability) #Dividing by p for Inverted Dropout
            return input_tensor * self.mask
        else:
            return input_tensor

    def backward(self, error_tensor):
        if self.testing_phase == False:
            return error_tensor*self.mask

        else:
            return error_tensor

