import numpy as np
from tensor import Tensor

class Sigmoid(Tensor):
    def __init__(self):
        pass
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def __call__(self, tensor):
        out = Tensor(self.sigmoid(tensor.value), operand=(tensor, ), operation='sigmoid', leaf=False)
        def sigmoid_grad_fxn():
            tensor.grad = self.sigmoid(tensor.value) * (1 - self.sigmoid(tensor.value)) * out.grad
        
        out.grad_fxn = sigmoid_grad_fxn
        return out
    
class Tanh(Tensor):
    def __init__(self):
        pass
    
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    
    def __call__(self, tensor):
        out = Tensor(self.tanh(tensor.value), operand=(tensor, ), operation='tanh', leaf=False)
        def tanh_grad_fxn():
            tensor.grad = (1 - (self.tanh(tensor.value)** 2)) * out.grad
        
        out.grad_fxn = tanh_grad_fxn
        return out
    
    
class Relu(Tensor):
    def __init__(self):
        pass
    
    def __call__(self, tensor):
        out = Tensor(max(0, tensor.value), operand=(tensor, ), operation='relu', leaf=False)
        def relu_grad_fxn():
            tensor.grad = (np.sign(tensor.value) + 1)/2 * out.grad
            
        out.grad_fxn = relu_grad_fxn
        return out
    
    
class Silu(Tensor):
    def __init__(self):
        self.sigmoid = Sigmoid()
    
    def __call__(self, tensor):
        sig = self.sigmoid(tensor.value)
        out = Tensor(tensor.value * sig, 
                     operand=(tensor, ), operation='silu', leaf=False)
        def silu_grad_fxn():
            tensor.grad = sig + tensor.value * sig * (1 - sig)
        
        out.grad_fxn = silu_grad_fxn
        return out
        