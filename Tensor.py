import numpy as np 
from functools import partialmethod 
# make  a decent numpy based NN library 
# then  add GPU support

class  Function:
    def __init__(self , *tensors):
        self.parents = tensors
        self.saved_tensors = []
    
    def save_for_backward(self, *tensors):  # tenzori *args  fazon
        self.saved_tensors.extend(tensors)

    def apply(self , func , *x): # arg je  add , mul itd <class '__main__.add'> | *x je tenzor
        ctx = func(self, *x) # func  je funkcija
        ret = Tensor(func.forward(ctx , self.data , *[t.data for  t  in x])) # unpack  without added dim
        
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

    # topo sort first
    def backward(self):
        if self._ctx is None: # ako je  leaf?
            return 
        assert self.shape == (1,) , f"Tensor must have a shape of (1,) instead of {self.shape}"
        # this  is  a tuple idk  if  it should be  that
        self.grad = np.ones_like(self.data)
        grads = self._ctx.backward(self._ctx , self.grad)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        #print(self._ctx.parents , self._ctx)
        for t, g in zip(self._ctx.parents , grads):
            if g.shape != t.shape:
                assert(False)
            t.grad = Tensor(g)
            t.backward()

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
        x,y  = ctx.saved_tensors # ovde  se cuvaju tenzori
        return x*out_grad , y*out_grad
register("mul",mul) 

class add(Function):
    @staticmethod 
    def forward(ctx,x,y):
        ctx.save_for_backward(x,y)
        return x+y
    @staticmethod
    def backward(ctx, out_grad):
        x,y  = ctx.saved_tensors
        return out_grad , out_grad
register("add",add)



t = Tensor([12])
t1 = Tensor([2])
c = t1.add(t)

#print(c.shape == (1,), type(c.shape) , c.shape)
c.backward()
print(t.grad, t1.grad)