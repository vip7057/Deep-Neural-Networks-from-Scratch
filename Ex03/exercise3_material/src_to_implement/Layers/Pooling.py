import numpy as np

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

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

        strided_output_shape = (input_tensor.shape[0], input_tensor.shape[1],
                     (input_tensor.shape[2] - self.pooling_shape[0]) // self.stride_shape[0] + 1,
                     (input_tensor.shape[3] - self.pooling_shape[1]) // self.stride_shape[1] + 1)

        self.pooling_output = np.array(self.pooling_output).reshape(strided_output_shape)
        self.max_index_array = np.array(self.max_index_array).reshape(*strided_output_shape,-1)
        #print(self.pooling_output.shape)
        return self.pooling_output

    def backward(self, error_tensor):
        error_tensor_next = np.zeros(self.input_tensor.shape)

        for b in range(error_tensor_next.shape[0]):
            for c in range(error_tensor_next.shape[1]):
                for j in range(error_tensor.shape[2]):
                    for i in range(error_tensor.shape[3]):
                        error_tensor_next[b,c,self.max_index_array[b,c,j,i,2],self.max_index_array[b,c,j,i,3]] += error_tensor[b,c,j,i]

        return error_tensor_next
        





