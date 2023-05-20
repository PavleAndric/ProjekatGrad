from projekatgrad.Tensor import  Tensor
import torch
import numpy as np
import unittest

def testing(shape, torhc_func , my_func, atol = 0 , rtol = 1e-6):
    torch.manual_seed(69) 

    torhc_tensors = [torch.rand(x) for  x  in shape]
    my_tensors = [Tensor(x.detach().numpy()) for x in torhc_tensors]
    desired = torhc_func(*torhc_tensors)
    input = my_func(*my_tensors)
    np.testing.assert_allclose(input.data , desired.detach().numpy() , atol = atol , rtol = rtol)

class TestOps(unittest.TestCase):

    def test_add(self):
        testing([(45, 65) ,(45, 65)] , torch.add, Tensor.Add)
    def test_sub(self):
        testing([(45, 65) ,(45, 65)] , torch.sub , Tensor.Sub)
    def test_mul(self):
        testing([(45, 65) ,(45, 65)] , torch.mul , Tensor.Mul)
    def test_pow(self):
        testing([(45, 65) ,(45, 65)] , torch.pow , Tensor.Pow)
    def test_div(self):
        testing([(45, 65) ,(45, 65)] , torch.div , Tensor.Div)
    def test_log(self):
        testing([(45, 65)] , torch.log , Tensor.Log)
    def test_relu(self):
        testing([(45 , 65)] , torch.relu  , Tensor.Relu)
    def test_tanh(self):
         testing([(45 , 65)] , torch.tanh  , Tensor.Tanh)
        
if __name__ == '__main__':
  unittest.main()  