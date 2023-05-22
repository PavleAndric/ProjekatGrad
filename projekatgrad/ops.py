from projekatgrad.Tensor import Function, register
import  numpy as  np

# dot  can be definig as  a conv 
# broadcasting is fckd ! 

def unbroad_reshape(x, desired_shape): # eithher  sum  or rehape
    
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
    def backward(ctx, out_grad): # TODO :make a nicer  way  of doing this
        x,y= ctx.saved_tensors
        t1 = np.expand_dims(x, 0) if x.ndim < 2 else x # za  Y.grad
        out_grad_1 = np.expand_dims(out_grad , 0) if out_grad.ndim < 2 else out_grad
        t2 = np.expand_dims(y  ,-1) if y.ndim < 2 else y
        out1 = (out_grad_1 @ t2.T).reshape(x.shape) # bad
        out2 = (t1.T @ out_grad_1).reshape(y.shape) # bad
        return  unbroad_reshape(out1 , x.shape) , unbroad_reshape(out2 ,y.shape)
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

class div(Function):
    def forward(ctx,x,y): 
        ctx.save_for_backward(x,y)
        return x/y
    @staticmethod
    def backward(ctx, out_grad):
        x,y  = ctx.saved_tensors
        return unbroad_reshape(1/y * out_grad , x.shape) , unbroad_reshape(-(x/y**2) * out_grad , y.shape) 
register("div",div)

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
    def backward(ctx, out_grad):
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
