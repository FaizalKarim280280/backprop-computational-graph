class Tensor:
    
    def __init__(self, 
                 value, 
                 operand=(), 
                 operation=None, 
                 leaf=True):
        
        self.value = value
        self.grad = 0
        self.operand = operand
        self.operation = operation
        self.grad_fxn = lambda: None
        self.gradients_calculated = False
        self.leaf = leaf
    
    def __repr__(self):
        return f"Tensor(value={self.value})"
    
    def __add__(self, tensor):
        out = Tensor(self.value + tensor.value, operand=(self, tensor), operation='+', leaf=False)
        def add_grad_fxn():
            self.grad += 1 * out.grad
            tensor.grad += 1 * out.grad
            
        out.grad_fxn = add_grad_fxn
        return out
    
    def __sub__(self, tensor):
        out = Tensor(self.value - tensor.value, operand=(self, tensor), operation='-', leaf=False)
        def sub_grad_fxn():
            self.grad += 1 * out.grad
            tensor.grad += -1 * out.grad
            
        out.grad_fxn = sub_grad_fxn
        return out
            
    def __mul__(self, tensor):
        out = Tensor(self.value * tensor.value, operand=(self, tensor), operation='*', leaf=False)
        def mul_grad_fxn():
            self.grad += tensor.value * out.grad
            tensor.grad += self.value * out.grad
            
        out.grad_fxn = mul_grad_fxn
        return out
    
    def __truediv__(self, tensor):
        out = Tensor(self.value / tensor.value, operand=(self, tensor), operation='/', leaf=False)
        def div_grad_fxn():
            self.grad += (1/tensor.value) * out.grad
            tensor.grad += -self.value /(tensor.value ** 2) * out.grad 
        
        out.grad_fxn = div_grad_fxn
        return out
    
    def __pow__(self, power):
        out = Tensor(self.value**power, operand=(self, ), operation=f'**{power}', leaf=False)
        def pow_grad_fxn():
            self.grad += power * (self.value ** (power - 1))
        
        out.grad_fxn = pow_grad_fxn
        return out    
            
    def backward(self):
        sorted, visited = [], set()
        
        def topological_sort(node):
            if node not in visited:
                visited.add(node)
                for v in node.operand:
                    topological_sort(v)
                sorted.append(node)

        topological_sort(self)
        for node in reversed(sorted):
            node.gradients_calculated = True
            node.grad_fxn()
            
    def reset_grads(self, ):
        visited = set()
        
        def dfs(node):
            if node not in visited:
                visited.add(node)
                node.grad = 0.0
                for v in node.operand:
                    dfs(v) 
                    
        dfs(self)   