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
                    assert ts.shape == gr.shape ,f"shapes of tensor {ts.shape} and grad {gr.shape} must be the same"
                    ts.grad = gr # gr is np.arr should  be a tensor?? 
            i._ctx = None

def register(name , func):
    partial = partialmethod(func.apply , func) 
    setattr(Tensor , name , partial) # setts new  attr to a Tensor

# dot  can be definig as  a conv 
# sub can be defined as a mul and add
# div can be defined with pow
# tanh can be definid as sigmoid
# softmax  implementation is just  stupid (find  a better way)

class dot(Function):
    @staticmethod
    def forward(ctx , x,y):
        out = x @ y 
        ctx.save_for_backward(x,y) 
        return out
         
    @staticmethod
    def backward(ctx, out_grad): # TODO :make a nicer  way  of doing this
        x,y= ctx.saved_tensors
        t1 = np.expand_dims(x, 0) if x.ndim < 2 else x # za  Y.grad
        out_grad_1 = np.expand_dims(out_grad , 0) if out_grad.ndim < 2 else out_grad
        t2 = np.expand_dims(y  ,-1) if y.ndim < 2 else y
        out1 = (out_grad_1 @ t2.T).reshape(x.shape) # bad
        out2 = (t1.T @ out_grad_1).reshape(y.shape) # bad
        return  out1 , out2
register("dot",dot)

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

class sub(Function):
    @staticmethod
    def forward(ctx,x,y):
        ctx.save_for_backward(x,y)
        return x-y
    @staticmethod
    def backward(ctx , out_grad):
        return out_grad , -out_grad
register("sub",sub)

class pow(Function):
    @staticmethod
    def forward(ctx , x , y):
        out = x**y 
        ctx.save_for_backward(x,y , out)
        return out
    @staticmethod
    def backward(ctx, out_grad):
        x,y,out= ctx.saved_tensors  # y can not be a tensor will couse  errors
        return y*(x**(y-1))*out_grad , np.log(x) * out_grad * out 
register("pow",pow)

class log(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.log(x)
    @staticmethod
    def backward(ctx, out_grad):
        x, = ctx.saved_tensors
        return 1 / x * out_grad
register("log",log)

class sum(Function):
    @staticmethod
    def forward(ctx , x):
        ctx.save_for_backward(x)
        return np.array([np.sum(x)])
    @staticmethod
    def backward(ctx , out_grad):
        x, = ctx.saved_tensors
        return np.ones_like(x) * out_grad
register("sum",sum)

# activations 
class relu(Function):
    @staticmethod
    def forward(ctx , x):
        ctx.save_for_backward(x)
        return np.maximum(0 , x)
    @staticmethod
    def backward(ctx , out_grad):
        x, = ctx.saved_tensors
        return np.greater(x,0) * out_grad
register("relu",relu)

class tanh(Function):
    @staticmethod
    def forward(ctx , x):
        out =  np.tanh(x)
        ctx.save_for_backward(out)
        return out
    @staticmethod
    def backward(ctx , out_grad):
        out, = ctx.saved_tensors
        return (1 - out**2) * out_grad
register("tanh",tanh)

class softmax(Function):
    @staticmethod
    def  forward(ctx , x):
        z = x - np.max(x)
        out = np.exp(z) / np.sum(np.exp(z))
        ctx.save_for_backward(x , out)
        return out
    @staticmethod

    def  backward(ctx , out_grad):
        x,out = ctx.saved_tensors
        chain = out_grad if out_grad.ndim < 2 else out_grad.reshape(-1 , 1) 
        chain = chain if x.ndim < 2 else chain.reshape(x.shape)
        output = (-np.outer(out , out) + np.diag(out.flatten())) @ chain
        output = output.reshape(x.shape)
        return  output
register("softmax",softmax)

