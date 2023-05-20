from projekatgrad.Tensor import Function, register
import  numpy as  np

# dot  can be definig as  a conv 
# tanh can be definid as sigmoid
# softmax  implementation is just stupid (find  a better way)

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