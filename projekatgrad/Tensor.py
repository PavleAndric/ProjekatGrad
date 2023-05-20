import numpy as np 
import math
from functools import partialmethod 
# make a decent numpy based NN library 
# then add GPU support

class  Function:
    def __init__(self , *tensors):
        self.parents = tensors
        self.saved_tensors = []
    
    def save_for_backward(self, *tensors):  # tenzori *args  fazon
        self.saved_tensors.extend(tensors)

    def apply(self , func , *x):
        
        ctx = func(self, *x)
        ret = Tensor(func.forward(ctx , self.data , *[t.data for  t  in x])) 
        ret._ctx = ctx
        return ret
    
class  Tensor:
    def __init__(self, data , requires_grad = False):
    
        self.grad ,self._ctx = None , None
        self.requires_grad  = requires_grad
        if isinstance(data, (list , tuple , int , float)):
            data = np.array(data , dtype = np.float32)
        if isinstance(data, np.ndarray): 
            data = data.astype(np.float32) 

        if not isinstance(data ,(np.ndarray , np.generic)):
            raise RuntimeError (f"Can't create a tensor from {data}")
        
        self.data = data 

    @property
    def dtype(self): return self.data.dtype
    @property 
    def shape(self): return self.data.shape 

    def __repr__(self): return f"Tensor {self.data} , {self.dtype}"

    def  toposort(self, visited = set() ,nodes = []):
        def topo(node):
            if node not in visited:
                visited.add(node)
                if node._ctx != None:
                    for x in node._ctx.parents:
                        topo(x)
                nodes.append(node)
            return nodes 
        return reversed(topo(self))

    def backward(self):
        if self._ctx is None: return 
        assert self.shape == (1,) , f"Tensor must have a shape of (1,) instead of {self.shape}"
        self.grad = np.ones(self.shape, dtype = np.float32) 

        for i in self.toposort():
            if i._ctx:
                grads = i._ctx.backward(i._ctx , i.grad) # example: if  _ctx is mul , it  will call i.mul.backward(set by register)
                if  not isinstance(grads, tuple): grads = [grads]
                for ts, gr in zip(i._ctx.parents, grads):
                    print(ts.shape, gr.shape)
                    assert ts.shape == gr.shape ,f"shapes of tensor {ts.shape} and grad {gr.shape} must be the same"
                    ts.grad = gr if ts.grad is None else (ts.grad + gr)
            i._ctx = None

    def Mul(self , x):
        return self.mul(x)
    def Add(self, x):
        return self.add(x)
    def Neg(self): 
        return self.mul(Tensor([-1]))
    def Sub(self,x): # 10 -5 = 5 10 * (5*-1)
        return self.add(x.mul(Tensor([-1])))  # this is ugly
    def Pow(self, x):
        return self.pow(x)
    def Div(self, x):
        return self.mul(x.pow(Tensor([-1])))
    def Log(self):
        return self.log()
    def Sqrt(self):
        return self.pow(Tensor[0.5])
    def Sum(self):
        return self.sum()
    def Tanh(self):
        return  self.tanh()
    def Relu(self):
        return self.relu()
    def Matmul(self,x):
        return  self.dot(x)
    def Softmax(self):
        return self.softmax()
    def Logsoftmax(self):
        return self.softmax().log()
    def test_softmax(self):
        pass
    
    
def register(name , func):
    partial = partialmethod(func.apply , func) 
    setattr(Tensor , name , partial) # setts new  attr to a Tensor

import projekatgrad.ops 