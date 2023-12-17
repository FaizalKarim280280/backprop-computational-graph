class Tensor:
    
    def __init__(self, value, operand=(), operation=None):
        self.value = value
        self.grad = None
        self.operand = operand
        self.operation = operation
    
    def __repr__(self):
        return f"Tensor(value={self.value})"
    
    def __add__(self, tensor):
        return Tensor(self.value + tensor.value, operand=(self, tensor), operation='+')
    
    def __sub__(self, tensor):
        return Tensor(self.value - tensor.value, operand=(self, tensor), operation='-')
    
    def __mul__(self, tensor):
        return Tensor(self.value * tensor.value, operand=(self, tensor), operation='*')
    
    def __truediv__(self, tensor):
        return Tensor(self.value / tensor.value, operand=(self, tensor), operation='/')
    
    def __pow__(self, tensor):
        return Tensor(self.value ** tensor.value, operand=(self, tensor), operation='^')
    
    
def main():
    a = Tensor(10)
    b = Tensor(2.5)
    c = Tensor(-5)
    
    d = a * b + c
    
    print(d)
    
if __name__ == "__main__":
    main()