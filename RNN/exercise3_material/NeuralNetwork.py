from tqdm import tqdm
from copy import deepcopy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        super().__init__()
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.data_layer = None
        self.loss_layer = None
        self.loss = []
        self.layers = []

    def append_layer(self, layer):
        if layer.trainable == True:
            layer.optimizer = deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            layer.testing_phase = False
            input_tensor = layer.forward(input_tensor)
        forward_out = input_tensor
        forward_loss = self.loss_layer.forward(forward_out, self.label_tensor)
        total_loss = forward_loss+self.optimizer.regularizer.norm(forward_loss) if self.optimizer.regularizer is not None else forward_loss
        return total_loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def train(self, iterations):
        for i in tqdm(range(iterations)):
            total_loss = self.forward()
            self.backward()
            self.loss.append(total_loss)

    def test(self, input_tensor):
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)
        predictions = input_tensor
        return predictions