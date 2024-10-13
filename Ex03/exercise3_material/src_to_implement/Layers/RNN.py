import numpy as np
from Exercise.Ex03.exercise3_material.src_to_implement.Layers import FullyConnected, TanH, Sigmoid

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(self.hidden_size)#Zero Initialization for Hidden States
        self.Wh = np.random.rand(input_size+hidden_size+1, hidden_size)#weight matrix for concatenated (input and hidden state) for tanH
        self.Why = np.random.rand(hidden_size+1,output_size)  # weight matrix for new hidden state for sigmoid
        self.Wh_shape = (self.Wh.shape[0]-1,self.Wh.shape[1])
        self.bh_shape = (1,self.Wh.shape[1])
        self.Why_shape = (self.Why.shape[0]-1,self.Why.shape[1])
        self.bhy_shape = (1, self.Why.shape[1])

        self.TanH = TanH.TanH()
        self.Sig = Sigmoid.Sigmoid()
        self.memory = False#If False ht = zeros, Else ht = ht-1
        self._optimizer= None

    @property
    def memorize(self):
        return self.memory
    @memorize.setter
    def memorize(self, memorize):
        self.memory= memorize

    #Added by me
    def initialize(self, weights_initializer, bias_initializer):
        #if (weights_initializer != None) and (bias_initializer != None):
        self.Wh = weights_initializer.initialize(self.Wh_shape,self.Wh_shape[0],self.Wh_shape[1])
        self.bh = bias_initializer.initialize(self.bh_shape ,self.bh_shape[0],self.bh_shape[1])
        self.Wh = np.concatenate((self.Wh, self.bh), axis=0)

        self.Why = weights_initializer.initialize(self.Why_shape, self.Why_shape[0], self.Why_shape[1])
        self.bhy = bias_initializer.initialize(self.bhy_shape, self.bhy_shape[0], self.bhy_shape[1])
        self.Why = np.concatenate((self.Why, self.bhy), axis=0)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        if self.memory is False:
            self.hidden_state = np.zeros(self.hidden_size)

        t = input_tensor.shape[0]#Batch dimension = time dimension

        #Creation of cache lists for storing values for backward pass
        output = np.zeros((t, self.output_size))  # Next Input Tensor (yT)
        fc2_out = []
        fc2_input =[]
        fc1_out = [] #ut
        fc1_input = []
        sigm = []
        tanh = []

        self.cache = {"fc2_out":fc2_out,"fc2_input":fc2_input,"fc1_out":fc1_out,"fc1_input":fc1_input, "sigm":sigm, "tanh":tanh}

        for i in range(t):
            #Concatenate the hidden state, input, and 1
            h_x_1_concatenated = np.vstack((self.hidden_state.reshape(self.hidden_size,1), input_tensor[i].reshape(input_tensor.shape[1],1),np.array([[1]])))
            tanH_input = h_x_1_concatenated.T@self.Wh #Fully Connected 1 Output
            self.hidden_state = self.TanH.forward(tanH_input) #New Hidden State
            tanh.append(self.TanH.tanh)


            #Append caches
            fc1_input.append(h_x_1_concatenated)
            fc1_out.append(tanH_input)

            # Concatenate the new hidden state and 1
            hnew_1_concatenated = np.vstack((self.hidden_state.reshape(self.hidden_size, 1),np.array([[1]])))
            sigm_input = hnew_1_concatenated.T @ self.Why  # Fully Connected 2 Output
            output[i] = self.Sig.forward(sigm_input)  # next input tensor for next RNN sequence.
            sigm.append(self.Sig.sigm)

            # Append caches
            fc2_input.append(hnew_1_concatenated)
            fc2_out.append(sigm_input)

        return output

    def backward(self, error_tensor):
        self.error_tensor= error_tensor
        t = error_tensor.shape[0] #batch dimension is the time dimension

        #Extract the Caches from Forward Pass
        fc2_input = self.cache["fc2_input"]
        fc1_input = self.cache["fc1_input"]
        sigm =  self.cache["sigm"]
        tanh = self.cache["tanh"]

        #Inittialize New Error Tensor
        error_tensor_new= np.zeros((t, self.input_size))

        # Initialize gradients
        self.dWhy = np.zeros_like(self.Why)
        self.dWh = np.zeros_like(self.Wh)
        self.dhnext = np.zeros_like(self.hidden_state)

        #Back Propagation Through Time
        for i in reversed(range(t)):
            # Backward pass through the Sigmoid layer
            self.Sig.sigm = sigm[i]
            dsigm_input = self.Sig.backward(self.error_tensor[i])
            self.dWhy += fc2_input[i]@dsigm_input

            # Backward pass through the Fully Connected 1 layer (TanH)
            self.TanH.tanh = tanh[i]
            dtanH_input = self.TanH.backward(self.dhnext + (dsigm_input @ self.Why[:-1, :].T)) #dhnext at time t = dsigm_input @ self.Why[:-1, :].T
            self.dWh += fc1_input[i]@dtanH_input

            # Update gradient for the next time step
            fc1_backward = dtanH_input @ self.Wh[:-1, :].T  # Output of FC1 bachwards dL/dconc.

            error_tensor_new[i] = np.squeeze(fc1_backward.T[self.hidden_size::])
            self.dhnext = np.squeeze(fc1_backward.T[0:self.hidden_size])

        #Weight Updates
        if self.optimizer is not None:
            self.Why = self.optimizer.calculate_update(self.Why, self.dWhy)
            self.Wh = self.optimizer.calculate_update(self.Wh, self.dWh)

        return error_tensor_new

    @property
    def gradient_weights(self):
        return self.dWh   #Considers the weights that are invovled in calculating the hidden state

    @property
    def weights(self):
        return self.Wh   #Getter property of weights

    @weights.setter
    def weights(self, w):
        self.Wh = w

    #Optimizer for regularization
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, o):
        self._optimizer = o

