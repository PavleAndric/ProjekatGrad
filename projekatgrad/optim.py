from projekatgrad.Tensor import  Tensor
import numpy  as  np
class Optimizer:

    def __init__(self , parameters):
        self.params = [x for x in  parameters]
        
    def zero_grad(self):
        for x in self.params:
           x.grad = 0.0

class SGD(Optimizer):

    def __init__(self, parameters , lr , momentum = 0.9):
        super().__init__(parameters)
        self.lr = lr
        self.m = momentum
        self.Vt = Tensor([0.])

    def step(self):

        for ts in self.params:
            ts -= self.lr * ts.grad 

        