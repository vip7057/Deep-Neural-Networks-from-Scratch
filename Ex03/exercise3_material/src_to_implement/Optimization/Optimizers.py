import numpy as np

class Optimizer:
    def __init__(self, regularizer = None):
        self.regularizer = regularizer

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
class Sgd(Optimizer):

    def __init__(self, learning_rate):
        super().__init__()
        self.lr = learning_rate

    def calculate_update(self,weight_tensor, gradient_tensor):
        constraint = self.lr * self.regularizer.calculate_gradient(weight_tensor) if (self.regularizer is not None) else 0
        w1 = weight_tensor - self.lr*gradient_tensor - constraint #wrt weights
        return w1


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.lr = learning_rate
        self.mr = momentum_rate
        self.v_old = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        v = self.mr*self.v_old - self.lr*gradient_tensor #exp. moving avg.
        self.v_old = v
        constraint = self.lr * self.regularizer.calculate_gradient(weight_tensor) if (self.regularizer is not None) else 0
        w1 = weight_tensor + v - constraint # wrt weights
        return w1

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.lr = learning_rate
        self.mu = mu
        self.rho = rho
        self.v_old = 0
        self.r_old = 0
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        v = (self.mu*(self.v_old)) + (1-self.mu)*gradient_tensor #exp. moving avg. 1
        r = (self.rho*(self.r_old)) + (1-self.rho)*(gradient_tensor**2) #exp. moving avg. 2
        self.v_old = v
        self.r_old = r
        self.k += 1
        v = v/(1-(self.mu**self.k))
        r = r/(1-(self.rho**self.k))

        constraint = self.lr * self.regularizer.calculate_gradient(weight_tensor) if (self.regularizer is not None) else 0
        w1 = weight_tensor - self.lr*v/(np.sqrt(r) + np.finfo(float).eps) - constraint
        return w1


