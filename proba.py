from mnist_dataset.make_mnist import fetch_mnist
import numpy as np
from projekatgrad.Tensor import  Tensor


X_train , Y_train , X_test , Y_test = fetch_mnist()
X_train = X_train / 255
X_train = X_train[:50].astype(np.float32)

epochs = 20

def one_hot(input):
    rom = np.zeros(10)
    rom[input] = 1
    return rom.reshape(1, 10)


class MyMnsit:
    
    def __init__(self):
        self.w_1 = Tensor(np.random.randn(784, 100) * 0.1)
        self.w_2 = Tensor(np.random.randn(100, 10)  * 0.1)
    
    def forward(self, x):
        return (x.dot(self.w_1)).Relu().dot(self.w_2).Softmax(1)
    

def one_hot(input):
    rom = np.zeros(10)
    rom[input] = 1
    return Tensor(rom.reshape(1, 10))

model = MyMnsit()

for ep in range(epochs):
    rom_loss = 0
    for x , y in  zip(X_train, Y_train):

        input = Tensor([x])
        target = one_hot(y)
        out = model.forward(input)
        loss = -(target  * out.Log()).Sum()
        rom_loss += loss.data
        
        loss.backward()

        model.w_1 = model.w_1 - (0.01 * model.w_1.grad)
        model.w_2 = model.w_2 - (0.01 * model.w_2.grad)
    
    print(rom_loss / 20)

import torch
class torchMnsit:
    
    def __init__(self):
        self.w_1 = torch.Tensor((np.random.randn(784, 100) * 0.1)) ;self.w_1.requires_grad= True
        self.w_2 = torch.Tensor((np.random.randn(100, 10)  * 0.1)) ;self.w_2.requires_grad= True
    
    def forward(self, x):
        out = (x.matmul(self.w_1)).relu().matmul(self.w_2).softmax(0)
        return out

epochs = 20

def one_hot(input):
    rom = torch.zeros(10)
    rom[input] = 1
    return rom.reshape(1, 10)

model = torchMnsit()
optim = torch.optim.SGD((model.w_1 , model.w_2) , lr = 0.01)

X_train_lol = torch.tensor(X_train)

for ep in range(epochs):
    rom_loss = 0
    for x , y in  zip(X_train_lol, Y_train):

        target = one_hot(y)
        out = model.forward(x)

        loss = -(target  * torch.log(out)).sum()
        rom_loss += loss.data

        optim.zero_grad()
        loss.backward()
        optim.step()

    print(rom_loss / 20)