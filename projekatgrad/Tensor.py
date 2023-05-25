import numpy as np 
from functools import partialmethod 
# make a decent numpy based NN library 
# then add GPU support 

class  Function:
    def __init__(self , *tensors):
        self.parents  = tuple([x for x in tensors if isinstance(x ,Tensor)]) # not  ideal
        #print(type(self.parents))
        self.saved_tensors = []
    
    def save_for_backward(self, *tensors):  
        self.saved_tensors.extend(tensors)

    def apply(self , func , *x):
        ctx = func(self, *x)
        ret = Tensor(func.forward(ctx , self.data  , *[t.data if isinstance(t, Tensor) else t for t in x])) # returns Tensor but args  are  np.array
        ret._ctx = ctx
        return ret
    
class  Tensor:
    def __init__(self, data , requires_grad = False):
    
        self.grad ,self._ctx = None , None
        self.requires_grad  = requires_grad # if any tensor  requres_grad then  do  backpass 
        if isinstance(data, (list , tuple , int , float)):
            data = np.array(data , dtype = np.float32)
        if isinstance(data, np.ndarray): 
            data = data.astype(np.float32) 

        if not isinstance(data ,(np.ndarray , np.generic)):
            raise RuntimeError (f"Can't create a tensor from {data , type(data)}")
        
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
                    assert ts.shape == gr.shape ,f"shapes of tensor {ts.shape} and grad {gr.shape} must be the same"
                    ts.grad = gr if ts.grad is None else (ts.grad + gr)
            i._ctx = None

    # maybe  not  ideal
    def assure_tensor(self , x, func =  None  , reversed = False):
          
        self = self if isinstance(self , Tensor) else Tensor(self)
        x =  x if isinstance(x , Tensor) else Tensor(x)
        return func(x, self) if reversed else func(self, x)      
   
    # fundamental
    def Mul(self, x  ,reversed = False):  return self.assure_tensor(x, Tensor.mul ,reversed)     
    def Add(self, x  ,reversed = False): return self.assure_tensor(x, Tensor.add ,reversed)
    def Pow(self, x , reversed = False): return self.assure_tensor(x, Tensor.pow ,reversed) 
    def Exp(self):  return self.exp() # assumes  self  is a tensor

    # basic math
    def Div(self, x , reversed = False): return self * x ** -1 if not reversed else x * self**-1 # this is  not  ideal
    def Sub(self,x , reversed = False): return self + (-x) if not reversed else x + (-self)
    def Log(self):return self.log()        
    def Sqrt(self): return self.pow(0.5)
    def Neg(self):  return self.mul(-1)
    def Matmul(self,x): return  self.dot(x) 
    
    def __add__(self, x): return self.Add(x)
    def __sub__(self, x): return self.Sub(x)
    def __mul__(self, x): return self.Mul(x)
    def __pow__(self, x): return self.Pow(x)
    def __truediv__(self, x): return self.Div(x)
    def __matmul__(self, x): return self.Matmul(x)
    def __neg__(self): return self.Neg()

    def __radd__(self, x): return self.Add(x , reversed = False)
    def __rsub__(self, x): return self.Sub(x , reversed = True)
    def __rmul__(self, x): return self.Mul(x , reversed = False)
    def __rpow__(self, x): return self.Pow(x , reversed = True)
    def __rtruediv__(self, x): return self.Div(x , reversed = True)
        
    # reduce
    def Sum(self, dim = None,  keepdims = False): return self.sum(dim , keepdims)

    # activations 
    def Sigmoid(self): return self._sig()
    def Relu(self): return self.relu() 
    def _sig(self): return 1 / (1 + (-self).exp())    
    def Tanh(self): return 2.0 * ((2.0 * self)._sig()) - 1.0
    def Logsoftmax(self, dim ): return self.Softmax(dim).log()   
        
    def Softmax(self , dim):
        exp  = self.exp()
        s = exp.sum(dim , keepdims = True if dim == 1 else False)
        out = exp / s 
        return out
    
def register(name , func): # this  is called n number of times (n  is tyhe number  of fucntins)
    partial = partialmethod(func.apply , func)
    setattr(Tensor , name , partial)

import projekatgrad.ops 