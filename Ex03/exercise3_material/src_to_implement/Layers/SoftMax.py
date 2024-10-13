
import numpy as np
from Exercise.Ex03.exercise3_material.src_to_implement.Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        input_tensor = np.exp(input_tensor - np.max(input_tensor))

        self.output_prob = input_tensor/np.sum(input_tensor, axis = 1, keepdims = True)
        return self.output_prob

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        errorSoftMax = self.output_prob*(error_tensor- np.sum(np.multiply(error_tensor, self.output_prob), axis=1, keepdims=True))
        return errorSoftMax

