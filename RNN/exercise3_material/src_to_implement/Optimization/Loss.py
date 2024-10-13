import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor
        output_loss = np.sum(label_tensor*(-np.log(prediction_tensor+np.finfo(float).eps)))
        return output_loss

    def backward(self, label_tensor):
        self.label_tensor = label_tensor
        backward_loss = -label_tensor/(self.prediction_tensor + np.finfo(float).eps) #error_tensor
        return backward_loss

         
