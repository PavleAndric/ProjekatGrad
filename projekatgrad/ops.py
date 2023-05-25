from projekatgrad.Tensor import Tensor, Function, register
import  numpy as  np

# dot  can be definig as  a conv(
# broadcasting is fckd ! 
# only  fundamental ops
def unbroad_reshape(x, desired_shape): # eithher  sum  or rehape
    print(x)
    if max(x.shape) > max(desired_shape):
        x = np.array([np.sum(x)])
        if x.shape != desired_shape:
            x = x.reshape(desired_shape)

    if x.shape != desired_shape:
        x  = x.reshape(desired_shape)
    return x

class dot(Function):
    @staticmethod
    def forward(ctx , x , y):
        out = x @ y 
        ctx.save_for_backward(x,y) 
        return out
    @staticmethod
    def backward(ctx, out_grad): 
        x,y= ctx.saved_tensors
        return  unbroad_reshape(out_grad @ y.T , x.shape) , unbroad_reshape(x.T @ out_grad ,y.shape)
register("dot",dot)

class mul(Function):
    @staticmethod 
    def forward(ctx,x,y): 
        ctx.save_for_backward(x,y)
        return x*y
    @staticmethod
    def backward(ctx, out_grad):
        x,y  = ctx.saved_tensors
        return unbroad_reshape(y * out_grad , x.shape) , unbroad_reshape(x * out_grad , y.shape)
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
        return unbroad_reshape(y*(x**(y-1))* out_grad  , x.shape), unbroad_reshape(np.log(x) * out_grad * out , y.shape)
register("pow",pow)

class exp(Function):
    @staticmethod
    def forward(ctx , x):
        out = np.exp(x)
        ctx.save_for_backward(out)
        return out
    @staticmethod
    def backward(ctx, out_grad):
        out , = ctx.saved_tensors # y can not be a tensor will couse  errors
        return unbroad_reshape(out * out_grad , out.shape)
register("exp",exp)

class log(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.log(x)
    @staticmethod
    def backward(ctx, out_grad):
        x, = ctx.saved_tensors
        return unbroad_reshape(1 / x * out_grad , x.shape)
register("log",log)

class sum(Function):
    @staticmethod
    def forward(ctx , x, dim = None, keepdims = False):
        ctx.save_for_backward(x)
        return np.array([np.sum(x)] if dim == None else np.sum(x, axis = dim , keepdims = keepdims))
     
    @staticmethod
    def backward(ctx , out_grad):
        x, = ctx.saved_tensors
        out = unbroad_reshape(np.ones_like(x) * out_grad , x.shape)
        return out  
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
