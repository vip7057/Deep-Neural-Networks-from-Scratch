import numpy as np

from Exercise.Ex03.exercise3_material.src_to_implement.Layers.Base import BaseLayer
class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.relu = np.maximum(0, input_tensor)
        self.mask = np.where(self.relu <= 0, 0, 1)
        return self.relu

    def backward(self, error_tensor):
        output = error_tensor*self.mask
        return output






