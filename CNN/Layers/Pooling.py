import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.ma.core import innerproduct
class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        new_shape = (input_tensor.shape[0], input_tensor.shape[1],
                     (input_tensor.shape[2] - self.pooling_shape[0]) // self.stride_shape[0] + 1,
                     (input_tensor.shape[3] - self.pooling_shape[1]) // self.stride_shape[1] + 1)
        op = np.zeros(new_shape)             

        self.pooling_output = []
        self.max_index_array = []

        for b in range(input_tensor.shape[0]):
            for c in range(input_tensor.shape[1]):
                for j in range(0, input_tensor.shape[2], self.stride_shape[0]):
                    for i in range(0, input_tensor.shape[3], self.stride_shape[1]):
                        if input_tensor[b, c, j:j+self.pooling_shape[0], i:i+self.pooling_shape[1]].shape == self.pooling_shape:
                            self.pooling_output.append(np.max(input_tensor[b, c, j:j+self.pooling_shape[0], i:i+self.pooling_shape[1]]))
                            element_number = np.argmax(input_tensor[b, c, j:j+self.pooling_shape[0], i:i+self.pooling_shape[1]])+1
                            max_y = np.ceil(element_number/self.pooling_shape[1]).astype(int)
                            max_x = element_number - (max_y-1)*self.pooling_shape[1]
                            max_index = [b,c,j+max_y-1, i+max_x-1]
                            self.max_index_array.append(max_index)

        self.pooling_output = np.array(self.pooling_output).reshape(new_shape)
        self.max_index_array = np.array(self.max_index_array).reshape(*new_shape,-1)
        #print(self.pooling_output.shape)
        return self.pooling_output

    def backward(self, error_tensor):
        output_error = np.zeros(self.input_tensor.shape)


        for b in range(output_error.shape[0]):
            for c in range(output_error.shape[1]):
                for j in range(error_tensor.shape[2]):
                    for i in range(error_tensor.shape[3]):
                        #print(self.max_index_array[b,c,j,i,2])
                        #print(self.max_index_array[b,c,j,i,3])
                        output_error[b,c,self.max_index_array[b,c,j,i,2],self.max_index_array[b,c,j,i,3]] += error_tensor[b,c,j,i]


        return output_error
"""


class Pooling:

    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = False
        self.input_value = None
        self.max_loc = None

    def forward(self, input_tensor):

        self.input_value = input_tensor

        # compute shape of the output
        new_shape = (input_tensor.shape[0], input_tensor.shape[1],
                     (input_tensor.shape[2] - self.pooling_shape[0]) // self.stride_shape[0] + 1,
                     (input_tensor.shape[3] - self.pooling_shape[1]) // self.stride_shape[1] + 1)

        output = np.zeros(new_shape)
        self.max_loc = []

        for b in range(new_shape[0]):
            for c in range(new_shape[1]):
                for j in range(new_shape[2]):
                    for i in range(new_shape[3]):
                        # compute max value
                        output[b, c, j, i] = np.amax(input_tensor[b, c,
                                                     j * self.stride_shape[0]:j * self.stride_shape[0] +
                                                                              self.pooling_shape[0],
                                                     i * self.stride_shape[1]:i * self.stride_shape[1] +
                                                                              self.pooling_shape[1]])

                        # index of max value
                        max_index = np.argwhere(input_tensor[b, c,
                                                j * self.stride_shape[0]:j * self.stride_shape[0] + self.pooling_shape[
                                                    0],
                                                i * self.stride_shape[1]:i * self.stride_shape[1] + self.pooling_shape[
                                                    1]] == output[b, c, j, i]) + [j * self.stride_shape[0],
                                                                                  i * self.stride_shape[1]]

                        max_index = np.array([b, c, max_index[0][0], max_index[0][1]])

                        # list of max locations
                        self.max_loc.append(max_index)

        return output

    def backward(self, error_tensor):
        output_error = np.zeros(self.input_value.shape)
        output_shape = output_error.shape
        error_shape = error_tensor.shape
        idx = 0
        for b in range(output_shape[0]):
            for c in range(output_shape[1]):
                for j in range(0, error_shape[2]):
                    for i in range(0, error_shape[3]):
                        # add error values to the max location
                        output_error[b, c, self.max_loc[idx][2], self.max_loc[idx][3]] += error_tensor[b, c, j, i]
                        idx = idx + 1

        return output_error"""

