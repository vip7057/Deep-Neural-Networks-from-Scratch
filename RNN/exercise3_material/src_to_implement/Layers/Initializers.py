import numpy as np

class Constant:
    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        init_weights = np.ones(weights_shape)*self.value
        return init_weights

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        init_weights = np.random.rand(*weights_shape)
        return init_weights

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_out + fan_in))
        init_weights = np.random.normal(0, sigma, size=weights_shape)
        return init_weights

class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/fan_in)
        init_weights = np.random.normal(0, sigma, size=weights_shape)
        return init_weights
