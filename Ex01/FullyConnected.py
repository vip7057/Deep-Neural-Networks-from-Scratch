from src_to_implement.Layers.Base import BaseLayer
import numpy as np
class FullyConnected(BaseLayer):

    def __init__(self,input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()
        self.trainable = True
        self.weights = np.random.rand(input_size+1,output_size)
        self._optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batchsize = input_tensor.shape[0]
        self.x = np.concatenate((input_tensor,np.ones((batchsize,1), dtype="int")), axis=1) #per batch 1 bias
        y = self.x@self.weights
        return  y

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


