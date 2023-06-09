from projekatgrad.Tensor import  Function, register
import  numpy as  np

def unbroad_reshape(x, desired_shape):
    axis = tuple(i for i ,x in enumerate(desired_shape) if x== 1 and x >= len(desired_shape)) if desired_shape != (1,) else None # None -> sum everting
    return x.sum(axis).reshape(desired_shape)

class dot(Function):
    @staticmethod
    def forward(ctx , x , y):
        out = x @ y 
        ctx.save_for_backward(x,y) 
        return out
    @staticmethod
    def backward(ctx, out_grad): 
        x,y= ctx.saved_tensors
        return  out_grad @ y.T , x.T @ out_grad
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
        x,y  = ctx.saved_tensors
        return unbroad_reshape(out_grad , x.shape) , unbroad_reshape(out_grad , y.shape)
register("add",add)

class pow(Function):
    @staticmethod
    def forward(ctx , x , y):
        out = x**y 
        ctx.save_for_backward(x,y , out)
        return out
    @staticmethod
    def backward(ctx, out_grad):
        x,y,out= ctx.saved_tensors  
        return unbroad_reshape(y*(x**(y-1))* out_grad  , x.shape), unbroad_reshape((np.log(x) * out_grad * out) , y.shape)
register("pow",pow)

class exp(Function):
    @staticmethod
    def forward(ctx , x):
        out = np.exp(x)
        ctx.save_for_backward(out)
        return out
    @staticmethod
    def backward(ctx, out_grad):
        out , = ctx.saved_tensors 
        return unbroad_reshape(out * out_grad , out.shape) # his  has to brodcast ? 
register("exp",exp)

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
    def forward(ctx , x, dim = None, keepdims = False):
        ctx.save_for_backward(x ,dim)
        out = np.array(np.sum(x , axis = dim , keepdims = keepdims))
        out  = out.reshape(1,) if out.shape  == () else out # tensor can't of dim 0
        return out
    @staticmethod
    def backward(ctx , out_grad):
        x, dim= ctx.saved_tensors
        f  = np.ones_like(x)
        out_grad = np.expand_dims(out_grad , dim if  dim is not None else ()) if len(f.shape) > len(out_grad.shape) else out_grad
        return f * out_grad  
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
