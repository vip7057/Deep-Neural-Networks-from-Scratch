from Exercise.Ex03.exercise3_material.src_to_implement.Layers.Base import BaseLayer
import numpy as np
class FullyConnected(BaseLayer):

    def __init__(self,input_size, output_size):
        self.fan_in = input_size
        self.fan_out = output_size
        super().__init__()
        self.trainable = True
        self.weights = np.random.rand(input_size+1,output_size)
        self._optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        weight_shape = (self.fan_in, self.fan_out)
        bias_shape = (1,self.fan_out) #check while unpacking
        self.weights = weights_initializer.initialize(weight_shape,self.fan_in,self.fan_out)
        self.bias = bias_initializer.initialize(bias_shape,1,self.fan_out)
        # combining weights and biases into weights
        self.weights = np.concatenate((self.weights, self.bias), axis=0)


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batchsize = input_tensor.shape[0]
        self.x = np.concatenate((input_tensor,np.ones((batchsize,1), dtype="int")), axis=1) #per batch 1 bias
        y = self.x@self.weights
        return y

    @property
    def optimizer(self):
        #Getter Property
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        #Setter property
        self._optimizer = opt

    @property
    def gradient_weights(self):
        return self.gradient_tensor

    def backward(self, error_tensor):
        x1 = self.x
        nobias_weights =  self.weights[0:-1,:]
        error_tensor_new = error_tensor@nobias_weights.T
        self.gradient_tensor = x1.T@error_tensor

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights,self.gradient_tensor)
        return error_tensor_new



