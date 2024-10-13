from copy import deepcopy

import numpy as np
from scipy import signal
from src_to_implement.Layers.Base import BaseLayer
class Conv(BaseLayer):
    strideshape = None
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        super().__init__()
        self.trainable = True
        self.weights = np.random.rand(num_kernels, *convolution_shape)
        self.bias = np.random.rand(num_kernels)
        self._optimizer = None

    @property
    def optimizer(self):
        #Getter Property
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        #Setter property
        opt1 = deepcopy(opt)
        opt2 = deepcopy(opt)
        self._optimizer = [opt1, opt2]

    @property
    def gradient_weights(self):
        return self.gradient_w

    @property
    def gradient_bias(self):
        return self.gradient_b

    def forward(self, input_tensor):
        self.input_tensor = input_tensor


        batch_size = input_tensor.shape[0]
        input_channels = input_tensor.shape[1]

        output_tensor = np.zeros((batch_size, self.num_kernels, *input_tensor.shape[2:]))
        for b in range(batch_size):
            for n_k in range(self.num_kernels):
                for c in range(input_channels):

                    output_tensor[b,n_k] += signal.correlate(input_tensor[b,c],self.weights[n_k,c], 'same')

                output_tensor[b,n_k] += self.bias[n_k]

        output_tensor = self.stride(output_tensor)
        return output_tensor

    def stride(self, output_tensor):

        if len(self.convolution_shape) > 2:
            output_tensor = output_tensor[:,:,0::self.stride_shape[0],0::self.stride_shape[1]]
            return output_tensor
        else:
            output_tensor = output_tensor[:, :, 0::self.stride_shape[0]]
            return output_tensor

    def padding(self, input_tensor):

        kernel_mn = np.array(self.convolution_shape[1:])
        padsize_1 = np.floor(kernel_mn/2).astype(int)
        padsize_2 = kernel_mn - padsize_1 - 1

        if len(kernel_mn)==2:
            padwidth = [(0,0),(0,0),(padsize_1[0], padsize_2[0]),(padsize_1[1], padsize_2[1])]
        else:
            padwidth = [(0, 0), (0, 0), (padsize_1[0], padsize_2[0])]

        input_tensor_padded =  np.pad(input_tensor, pad_width= padwidth, constant_values=0)
        return input_tensor_padded


    def stride_upsample(self, error_tensor, input_shape):

        result = np.zeros(input_shape)
        #if len(self.convolution_shape) > 2:  # 2D
        if len(input_shape) > 1:  # 2D
            result[0::self.stride_shape[0], 0::self.stride_shape[1]] = error_tensor
            return result
        else:
            result[0::self.stride_shape[0]] = error_tensor
            return result


    def backward(self,error_tensor):
        self.gradient_w = np.zeros(self.weights.shape)
        if len(self.convolution_shape) > 2:
            axis = (0, 2, 3)
        else:
            axis = (0, 2)
        self.gradient_b = np.sum(error_tensor, axis=axis)
        new_error_tensor = np.zeros(self.input_tensor.shape)
        input_depth = self.input_tensor.shape[1]
        padded_input = self.padding(self.input_tensor)

        for b in range(self.input_tensor.shape[0]):
            for n_k in range(self.num_kernels):
                for c in range(input_depth):
                    # upsample the error tensor according to stride
                    upsampled_error = self.stride_upsample(error_tensor[b, n_k], self.input_tensor.shape[2:])
                    out_w = signal.correlate(padded_input[b, c], upsampled_error, "valid")
                    self.gradient_w[n_k, c] += out_w
                    # padd the upsampled error

                    out_err = signal.convolve(upsampled_error, self.weights[n_k, c], "same")
                    new_error_tensor[b, c] += out_err

        if (self.optimizer != None):
            self.weights = self.optimizer[0].calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer[1].calculate_update(self.bias, self.gradient_bias)

        return new_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)





