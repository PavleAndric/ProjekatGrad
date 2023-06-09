import sys
import os
#import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from mnist_dataset.fetch_mnist import fetch_mnist
from  projekatgrad.Tensor import  Tensor
from projekatgrad.optim import SGD ,Adam

X_train , Y_train , X_test , Y_test = fetch_mnist()
X_train = X_train / 255
rombcina = 25
X_train = X_train[:rombcina].astype(np.float32)

epochs = 20

def one_hot_my(input):
    rom = np.zeros(10)
    rom[input] = 1
    return Tensor(rom.reshape(1, 10))

class MyMnsit:
    
    def __init__(self):
        self.w_1 = Tensor(np.random.randn(784, 100) * 0.1)
        self.w_2 = Tensor(np.random.randn(100, 10)  * 0.1)
    
    def forward(self, x):
        return (x.dot(self.w_1)).Relu().dot(self.w_2).Softmax(1)
    
model = MyMnsit()
optim = Adam([model.w_1  , model.w_2] , 0.001 , (0.9 ,0.999))

for ep in range(epochs):
    rom_loss = 0
    for x , y in  zip(X_train, Y_train):

        input = Tensor([x]) 
        target = one_hot_my(y)

        out = model.forward(input)

        loss = (target  * out.Log()).Sum() * -1
        rom_loss += loss.data
        optim.zero_grad()
        loss.backward() 
        optim.step()

    print(rom_loss / rombcina)

import torch

class torchMnsit:
    
    def __init__(self):
        self.w_1 = torch.Tensor((np.random.randn(784, 100) * 0.1)) ;self.w_1.requires_grad= True
        self.w_2 = torch.Tensor((np.random.randn(100, 10)  * 0.1)) ;self.w_2.requires_grad= True
    
    def forward(self, x):
        out = (x.matmul(self.w_1)).relu().matmul(self.w_2).softmax(0)
        return out

def one_hot(input):
    rom = torch.zeros(10)
    rom[input] = 1
    return rom.reshape(1, 10)

model1 = torchMnsit()
optim1  = torch.optim.Adam((model1.w_1 , model1.w_2) , lr = 0.001 , betas = (0.9 ,0.999))

X_train_lol = torch.tensor(X_train)

for ep in range(epochs):
    rom_loss = 0
    for x , y in  zip(X_train_lol, Y_train):

        target = one_hot(y)
        out = model1.forward(x)

        loss = -(target  * torch.log(out)).sum()
        rom_loss += loss.data

        optim1.zero_grad()
        loss.backward()
        optim1.step()
    print(rom_loss / rombcina)

print(len(optim.params))