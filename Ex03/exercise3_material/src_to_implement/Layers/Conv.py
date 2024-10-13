from copy import deepcopy
import numpy as np
import scipy
from scipy import signal
from Exercise.Ex03.exercise3_material.src_to_implement.Layers.Base import BaseLayer
class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        super().__init__()
        self.trainable = True
        self.weights = np.random.rand(num_kernels, *convolution_shape)
        self.bias = np.random.rand(num_kernels)
        self._optimizer = None
        self.input_size = np.prod(convolution_shape)
        self.output_size = np.prod(convolution_shape[1:])*self.num_kernels

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        opt1 = deepcopy(opt)
        opt2 = deepcopy(opt)
        self._optimizer = [opt1, opt2]

    @property
    def gradient_weights(self):
        return self.dw

    @property
    def gradient_bias(self):
        return self.db

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)
        self.bias = bias_initializer.initialize(self.bias.shape, self.input_size, self.output_size)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        input_channels = input_tensor.shape[1]
        output_tensor = np.zeros((batch_size, self.num_kernels, *input_tensor.shape[2:]))

        for b in range(batch_size):
            for n_k in range(self.num_kernels):
                #for c in range(input_channels):
                    #output_tensor[b,n_k] += signal.correlate(input_tensor[b,c],self.weights[n_k,c], 'same')
                    #output_tensor[b,n_k] += signal.correlate(input_tensor[b,c],self.weights[n_k,c], 'same')
                output_tensor[b, n_k] += signal.correlate(input_tensor[b], self.weights[n_k], 'same')[input_channels//2]
                output_tensor[b,n_k] += self.bias[n_k]

        output_tensor = self.stride(output_tensor)
        return output_tensor

    def backward(self,error_tensor):
        self.dw = np.zeros(self.weights.shape)
        error_tensor_next = np.zeros(self.input_tensor.shape)
        input_channels = self.input_tensor.shape[1]
        # Reshape and transpose the weight tensor
        reshaped_weight_tensor = self.weights.transpose((1, 0, 2, 3)) if len(self.weights.shape)>3 else self.weights.transpose((1, 0, 2))

        reshaped_error_tensor = np.zeros((error_tensor.shape[0], error_tensor.shape[1], *self.input_tensor.shape[2:]))
        if len(self.input_tensor.shape) > 3:
            reshaped_error_tensor[:, :, 0::self.stride_shape[0], 0::self.stride_shape[1]] = error_tensor
        else:
            reshaped_error_tensor[:,:,0::self.stride_shape[0]] = error_tensor

        self.db = np.sum(error_tensor, axis=(0,2,3) if len(self.convolution_shape) > 2 else (0,2))

        for b in range(self.input_tensor.shape[0]):
            for n_k in range(self.num_kernels):
                for c in range(input_channels):
                    self.dw[n_k,c] += signal.correlate(self.padding(self.input_tensor)[b,c], reshaped_error_tensor[b,n_k], "valid")
                    error_tensor_next[b, c] = signal.correlate(reshaped_error_tensor[b], np.flip(reshaped_weight_tensor[c], axis=(1,2)), mode='same')[(self.num_kernels)//2] if len(self.weights.shape)>3\
                        else signal.correlate(reshaped_error_tensor[b], np.flip(reshaped_weight_tensor[c], axis=(1)), mode='same')[(self.num_kernels)//2]

        if (self.optimizer is not None):
            self.weights = self.optimizer[0].calculate_update(self.weights, self.dw)
            self.bias = self.optimizer[1].calculate_update(self.bias, self.db)

        return error_tensor_next

    def stride(self, output_tensor):
        return output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] if len(self.convolution_shape) > 2 else output_tensor[:, :, ::self.stride_shape[0]]

    def padding(self, input_tensor):
        kernel_mn = np.array(self.convolution_shape[1:])

        # pad widths for each dimension
        padwidth = [(0, 0) for _ in range(len(input_tensor.shape))]

        for i in range(len(kernel_mn)):
            padsize_1 = np.floor(kernel_mn[i] / 2).astype(int)
            padsize_2 = kernel_mn[i] - padsize_1 - 1
            padwidth[i + 2] = (padsize_1, padsize_2)
        # Apply padding
        input_tensor_padded = np.pad(input_tensor, pad_width=padwidth, constant_values=0)

        return input_tensor_padded













