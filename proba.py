#from projekatgrad.Tensor import  Tensor
import  math
import numpy  as  np

from torch.nn.functional import one_hot

#
#class Tnet():
#
#    def __init__(self):
#        self.L1 = torch.randn(784, 128) * math.sqrt(1/128)   ;self.L1.requires_grad = True
#        self.L2 = torch.randn(128, 10)  * math.sqrt(1/10)    ;self.L1.requires_grad = True
#
#    def forward(self, x):
#        o = torch.nn.ReLU()
#        
#        out = o(x.matmul(self.L1)).matmul(self.L2).softmax(dim = 0)
#        print(out.shape)
#        return out
#
#input =  torch.randn(1, 784)    
#model = Tnet()
#rom = model.forward(input)
#

np.random.seed(100)
t1 = np.random.randn(1,2).astype(np.float32)
t2 = np.random.randn(2,2).astype(np.float32)
t3 = np.random.randn(2  , 10).astype(np.float32)



#from tinygrad.tensor import  Tensor
#input_tensor = Tensor(t1 ,requires_grad = True)
#w_1 = Tensor(t2,requires_grad = True)
#w_2 = Tensor(t3,requires_grad = True)
#t1 = input_tensor @ w_1
#t3 = t1 @ w_2
#t4 = t3.softmax(axis= 0)
#t5 = t4.sum()
#t5.backward()
#print(t4.numpy())




from projekatgrad.Tensor import  Tensor
input_tensor = Tensor(t1)
w_1 = Tensor(t2)
w_2 = Tensor(t3)
t1 = input_tensor @ w_1
t3 = t1 @ w_2
t4 = t3.Softmax(dim = 1)
t5 = t4.Sum()
t5.backward()
print(t4)



#tiny_grad()
