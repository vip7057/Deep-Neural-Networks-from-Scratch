import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.lamda = alpha

    def calculate_gradient(self, weights):#For Backpropagation
        output = self.lamda*weights
        return output
    def norm(self, weights):#For Forwardpropagation(Total Loss = Cross Entropy Loss + Regularizer Loss)
        l2_loss_per_layer = self.lamda * np.linalg.norm(weights.flatten())**2
        return l2_loss_per_layer



class L1_Regularizer:
    def __init__(self, alpha):
        self.lamda = alpha

    def calculate_gradient(self, weights):#For Backpropagation
        output = self.lamda * np.sign(weights)
        return output

    def norm(self, weights):#For Forwardpropagation(Total Loss = Cross Entropy Loss + Regularizer Loss)
        l1_loss_per_layer = self.lamda*np.sum(np.absolute(weights.flatten()))
        return l1_loss_per_layer
        


