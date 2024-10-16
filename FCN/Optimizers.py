class Sgd(float):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self,weight_tensor, gradient_tensor):
        updated_weights = weight_tensor - self.learning_rate*gradient_tensor #wrt weights
        return updated_weights

