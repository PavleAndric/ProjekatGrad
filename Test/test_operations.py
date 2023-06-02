from projekatgrad.Tensor import  Tensor
import torch
import numpy as np
import unittest

def helper_test(shape, torhc_func , my_func, atol = 0 , rtol = 1e-6):
    
    torhc_tensors = [torch.rand(x) for  x  in shape] # torch_Tenors
    my_tensors = [Tensor(x.detach().numpy()) for x in torhc_tensors] # my_tensors wtih  same elements
    desired = torhc_func(*torhc_tensors)
    input = my_func(*my_tensors)
    np.testing.assert_allclose(input.data , desired.detach().numpy() , atol = atol , rtol = rtol)

class TestOps(unittest.TestCase):
    # test fundamental math
    def test_add(self):
        helper_test([(45, 65) ,(45, 65)] ,   lambda x ,y : x + y ,  lambda x ,y : x + y )
        helper_test([(45, 65)]  , lambda x: x+10  , lambda x: x+10)
        helper_test([(45, 65)]  , lambda x: 10+x  , lambda x: 10+x)
    def test_sub(self):
        helper_test([(45, 65) ,(45, 65)] ,  lambda x ,y : x - y  ,  lambda x ,y : x - y )
        helper_test([(45, 65)]  , lambda x: x-10  , lambda x: x-10)
        helper_test([(45, 65)]  , lambda x: 10-x  , lambda x: 10-x)
    def test_mul(self):
        helper_test([(45, 65) ,(45, 65)] ,  lambda x ,y : x * y  ,   lambda x ,y : x * y )
        helper_test([(45, 65)]  , lambda x: x*0.3  , lambda x: x*0.3)
        helper_test([(45, 65)]  , lambda x: 0.3*x  , lambda x: 0.3*x)
    def test_pow(self):
        helper_test([(45,64) ,(45,64)] , lambda x ,y  : x.pow(y) ,  lambda x ,y  : x.Pow(y) )
        helper_test([(45,64)] , lambda x: x**2 , lambda x: x**2)
        helper_test([(45,64)] , lambda x: x**-2 , lambda x: x**-2)
    def test_div(self):
        helper_test([(45, 65) ,(45, 65)] , lambda x ,y: x / y , lambda x ,y: x / y)
        helper_test([(45,64)] , lambda x: x/2 , lambda x: x/2)
        helper_test([(45,64)] , lambda x: 2/x , lambda x: 2/x)
    def test_matmul(self):
        helper_test([(45, 65) ,(65, 45)] , lambda x ,y : x @ y , lambda x ,y : x @ y)
        helper_test([(45, 65) ,(65, 1)] , lambda x ,y : x @ y , lambda x ,y : x @ y)
        helper_test([(45, 1 ) ,(1, 1)] , lambda x ,y : x @ y , lambda x ,y : x @ y)
    # sum  and 
    def test_sum(self):
        helper_test([(45 , 65)] , lambda x: x.sum() , lambda x: x.Sum())
        helper_test([(45 , 65)] , lambda x : x.sum(dim = 0)   , lambda x:x.Sum(dim = 0))
        helper_test([(45 , 65)] , lambda x : x.sum(dim = 1), lambda x:x.Sum(dim = 1))
        helper_test([(45 , 65)] , lambda x : x.sum(dim = 1,keepdims = True)  , lambda x:x.Sum(dim = 1 ,keepdims = True))
    def test_epx(self):
        helper_test([(45 , 65)] , lambda x: x.exp()  , lambda x: x.Exp())
    # test  unary  functions
    def test_neg(self):
        helper_test([(45, 65)] , lambda x :-x , lambda x:-x)
    def test_sqrt(self):
        helper_test([(45, 65)] , lambda x :torch.sqrt(x) , lambda x : x.Sqrt())
    def test_log(self):
        helper_test([(45, 65)] , lambda x: x.log() , lambda x : x.Log())
    #test activations
    def test_relu(self):
        helper_test([(45 , 65)] , lambda x: x.relu(),lambda x: x.Relu() )
    def test_tanh(self):
         helper_test([(45 , 65)] , lambda x: x.tanh() , lambda x: x.Tanh(), atol=1e-6 , rtol=1e-6)
    def test_sigmoid(self):
        helper_test([(45 , 65)] , lambda x: x.sigmoid() ,lambda x: x.Sigmoid())
    def test_softmax(self):
        helper_test([(45 , 65)] , lambda x:x.softmax(dim = 1) , lambda x: x.Softmax(dim = 1))
        helper_test([(45 , 65)] , lambda x:x.softmax(dim = 0) , lambda x: x.Softmax(dim = 0))
    def test_logsoftmax(self):
        helper_test([(45 , 65)] , lambda x:x.log_softmax(dim = 1) , lambda x: x.Logsoftmax(dim = 1))
        helper_test([(45 , 65)] , lambda x:x.log_softmax(dim = 0) , lambda x: x.Logsoftmax(dim = 0))

if __name__ == '__main__':
  unittest.main()  