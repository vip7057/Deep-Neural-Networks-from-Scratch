import numpy as np

class Sgd(float):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self,weight_tensor, gradient_tensor):
        updated_weights = weight_tensor - self.learning_rate*gradient_tensor #wrt weights
        return updated_weights


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v_old = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        v = self.momentum_rate*self.v_old - self.learning_rate*gradient_tensor
        updated_weights = weight_tensor + v  # wrt weights
        self.v_old = v
        return updated_weights

class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.k = 1
        self.v_old = 0
        self.r_old = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        #first order and second order gradient descent

        v = (self.mu*(self.v_old)) + (1-self.mu)*gradient_tensor
        r = (self.rho*(self.r_old)) + (1-self.rho)*(gradient_tensor**2)
        self.v_old = v
        self.r_old = r

        #bias correction
        v = v/(1-(np.power(self.mu,self.k)))
        r = r/(1-(np.power(self.rho,self.k)))
        self.k += 1

        #weight update
        updated_weights = weight_tensor - self.learning_rate*v/(np.sqrt(r) + np.finfo(float).eps)

        return updated_weights

