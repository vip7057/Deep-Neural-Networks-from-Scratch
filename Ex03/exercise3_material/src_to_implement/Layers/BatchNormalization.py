import numpy as np
from Exercise.Ex03.exercise3_material.src_to_implement.Layers.Helpers import compute_bn_gradients
from Exercise.Ex03.exercise3_material.src_to_implement.Layers.Base import BaseLayer
from copy import deepcopy

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.decay = 0.8
        self.mean = np.zeros(self.channels, dtype=float)
        self.var = np.zeros(self.channels, dtype=float)
        self.moving_mean = np.zeros(self.channels)
        self.moving_var = np.zeros(self.channels)
        self.eps = 1e-11
        self.initialize(None, None)
        self._optimizer = None
        self.testing_phase = False
        self.cnn = False

    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        opt1 = deepcopy(opt)
        opt2 = deepcopy(opt)
        self._optimizer = [opt1, opt2]

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones(self.channels)#Gamma
        self.bias = np.zeros(self.channels) #Beta

    
    def reformat(self, tensor):#Reformating for CNN case
        if len(tensor.shape) == 4:  #4D to 2D
            self.Batch, self.Channel, self.M, self.N = tensor.shape
            tensor = tensor.reshape(self.Batch, self.Channel, self.M * self.N)
            tensor = np.transpose(tensor, (0, 2, 1))#Changing Axes
            tensor = tensor.reshape(-1, self.Channel)
        else:  #2D to 4D
            tensor = tensor.reshape(self.Batch, self.M * self.N, self.Channel)
            tensor = np.transpose(tensor, (0, 2, 1))
            tensor = tensor.reshape(self.Batch, self.Channel, self.M, self.N)
        return tensor


    def forward(self, input_tensor):
        if len(input_tensor.shape) == 4:
            input_tensor = self.reformat(input_tensor)  # 4D to 2D for cnn
            self.cnn = True

        self.input_tensor = input_tensor

        if self.testing_phase == False:#Training Phase
            self.mean = np.mean(input_tensor, axis=0)
            self.var = np.var(input_tensor, axis=0)

            # Normalize Batch Input For Training Case
            self.x_tilde = (input_tensor - self.mean) / np.sqrt(self.var + self.eps)
            output = self.weights * self.x_tilde + self.bias

            # Moving_average_estimation during training

            #Initialize Moving mean and var with first batch mean and var.
            if np.all(self.moving_mean == 0) and np.all(self.moving_var == 0):
                self.moving_mean = self.mean
                self.moving_var = self.var

            #Calculate moving average update
            else:
                self.moving_mean = self.decay * self.moving_mean + (1.0 - self.decay) * self.mean
                self.moving_var = self.decay * self.moving_var + (1.0 - self.decay) * self.var

        else:#Testing Phase

            #Normalize Input for Testing Phase
            self.x_tilde = (input_tensor - self.moving_mean.copy()) / np.sqrt(self.moving_var.copy() + self.eps)
            output = self.weights * self.x_tilde + self.bias

        if self.cnn: # Reformat to 4D for cnn
            output = self.reformat(output)

        return output

    def backward(self, error_tensor):
        if self.cnn:#Backward Case for CNN
            error_tensor = self.reformat(error_tensor)

        self.gradient_weights = np.sum(error_tensor * self.x_tilde, axis=0)#dGamma
        self.gradient_bias = np.sum(error_tensor, axis=0)#dBeta
        if (self._optimizer != None):
            self.weights = self._optimizer[0].calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer[1].calculate_update(self.bias, self.gradient_bias)

        input_gradient = compute_bn_gradients(error_tensor,self.input_tensor,self.weights,self.mean,self.var)#dX

        if self.cnn:#Reformat to 4D for cnn
            input_gradient = self.reformat(input_gradient)

        return input_gradient
