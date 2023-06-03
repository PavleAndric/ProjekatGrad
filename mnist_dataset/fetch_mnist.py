import os 
import numpy as np
import  gzip

def parse(file):
    return np.frombuffer(gzip.open(file).read() , dtype=np.uint8)

def fetch_mnist():
    
    X_train = parse(os.path.dirname(__file__)+"/train-images-idx3-ubyte.gz")[0x10:].reshape((-1 , 28*28))
    Y_train = parse(os.path.dirname(__file__)+"/train-labels-idx1-ubyte.gz")[8:]
    X_test = parse(os.path.dirname(__file__)+"/t10k-images-idx3-ubyte .gz")[0x10:].reshape(-1 , 28*28)
    Y_test = parse(os.path.dirname(__file__)+"/t10k-labels-idx1-ubyte.gz")[8:]
    return X_train , Y_train , X_test , Y_test
