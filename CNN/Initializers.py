import numpy as np

class Constant:
    def __init__(self, const_value = 0.1):
        self.const_value = const_value

    def initialize(self, weights_shape, fan_in, fan_out):
        init_weights = np.full(weights_shape, self.const_value)
        return init_weights


class UniformRandom:
    def __init__(self):
        pass
    def initialize(self, weights_shape, fan_in, fan_out):
        init_weights = np.random.rand(weights_shape[0],weights_shape[1])
        return init_weights

class Xavier:
    def __init__(self):
        pass
    def initialize(self, weights_shape, fan_in, fan_out):
        variance = np.sqrt(2/(fan_in+fan_out))
        init_weights = np.random.normal(0, variance, size = weights_shape)
        return init_weights
class He:
    def __init__(self):
        pass
    def initialize(self, weights_shape, fan_in, fan_out):
        variance = np.sqrt(2 / fan_in)
        init_weights = np.random.normal(0, variance, size=weights_shape)
        return init_weights
