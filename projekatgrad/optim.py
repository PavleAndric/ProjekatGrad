from projekatgrad.Tensor import  Tensor
import numpy  as  np
class Optimizer:

    def __init__(self , parameters ,lr):
        self.params = [x for x in  parameters]
        self.lr = lr
        
    def zero_grad(self):
        for x in self.params:
           x.grad = None # maybe should  be 0

class SGD(Optimizer):

    def __init__(self, parameters , lr , momentum = 0.0 , nestorov = False):
        super().__init__(parameters , lr)
        self.m = momentum ; self.n = nestorov
        self.Vt = [Tensor([0.]) for x in self.params] 

    def step(self):

        for i , t in enumerate(self.params):

            grad = t.grad + self.Vt[i] if  self.n else t.grad    
            self.Vt[i] = self.m  * self.Vt[i] + (self.lr * grad)
            t.assign(t - self.Vt[i])
            
class Adam(Optimizer):
    def  __init__(self, parameters , lr = 0.001 ,betas = (0.9 ,0.999)):
        super().__init__(parameters, lr)
        self.b1 , self.b2 = betas
        self.e = 1e-8
        self.Vt = [Tensor([0.]) for x in self.params]
        self.Mt = [Tensor([0.]) for x in self.params]
        self.t = 0

    def step (self): # super slow
        self.t += 1    
        for i , t in enumerate(self.params):
            self.Mt[i] = self.b1 * self.Mt[i] +  (1 - self.b1) * t.grad
            self.Vt[i] = self.b2 * self.Vt[i] +  (1 - self.b2) * t.grad **2
            m_hat = self.Mt[i] / (1 - (self.b1 ** self.t))
            v_hat = self.Vt[i] / (1 - (self.b2 ** self.t))
            t.assign(t  - self.lr *  (m_hat / (v_hat.Sqrt() + self.e)))



