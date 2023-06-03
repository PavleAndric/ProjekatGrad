from projekatgrad.Tensor import  Tensor
import  numpy  as np

w_1 = Tensor(np.random.randn(784, 12) * 0.1)
w_2 = Tensor(np.random.randn(12, 10)  * 0.1)


def forward(x):
        return (x.dot(w_1)).Relu().dot(w_2).Softmax(0)

x = np.random.randn(1, 784)
t = Tensor(x)


for i in range(10):
    input = Tensor(t)
    out = forward(input)
    loss = out.Sum()
    print(len(list(loss.toposort())))

print()
from tinygrad.tensor import  Tensor

w_1 = Tensor.randn(784, 12 , requires_grad = True)
w_2 = Tensor.randn(12, 10 , requires_grad = True)


def forward(x):
        return (x.dot(w_1)).relu().dot(w_2).softmax(0)

t = Tensor.randn(1, 784)


for i in range(10):
    
    out = forward(t)
    loss = out.sum()
    #loss.backward()
    #print(w_1.grad.numpy().mean())
    print(len(loss.deepwalk()))
