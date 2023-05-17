import numpy as np 
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
        ctx = func(self, *x)  # func je(mul , add itd)
        ret = Tensor(func.forward(ctx , self.data , *[t.data for  t  in x])) 
        ret._ctx = ctx
        return ret

class  Tensor:

    def __init__(self, data , requires_grad = False):
        
        if isinstance(data, (list , tuple , int , float)):
            data = np.array(data , dtype = np.float32)
        if isinstance(data, np.ndarray): 
            data = data.astype(np.float32) 

        if not isinstance(data , np.ndarray):
            raise RuntimeError (f"Can't create a tensor from {data}")
        self.data = data 
        self.grad = None
        self.requires_grad  = requires_grad

        self._ctx = None 

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
        return topo(self)

    def backward(self):
        # must modify for unary functions
        if self._ctx is None: return 
        assert self.shape == (1,) , f"Tensor must have a shape of (1,) instead of {self.shape}"
        self.grad = np.ones(self.shape, dtype = np.float32)

        for i in reversed(self.toposort()):
            if i._ctx != None:
                grads = i._ctx.backward(i._ctx , i.grad)
                for ts, gr in zip(i._ctx.parents, grads):
                    assert ts.shape == gr.shape ,f"shapes of tensor {ts.shape} and grad {gr.shape} must be the same"
                    ts.grad = gr

def register(name , fxn):
    partial = partialmethod(fxn.apply , fxn) #
    setattr(Tensor , name , partial)
# partialmethod(fnx.apply) # hoces  da  aplajujes  na odredjenu funkciju(add , mul , itd)
    
class mul(Function):
    @staticmethod 
    def forward(ctx,x,y): 
        ctx.save_for_backward(x,y)
        return x*y
    @staticmethod
    def backward(ctx, out_grad):
        x,y  = ctx.saved_tensors
        return y*out_grad , x*out_grad 
register("mul",mul) 

class add(Function):
    @staticmethod 
    def forward(ctx,x,y):
        ctx.save_for_backward(x,y)
        return x+y
    @staticmethod
    def backward(ctx, out_grad):
        return out_grad , out_grad
register("add",add)
