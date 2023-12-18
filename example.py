from tensor import Tensor
from plot_graph import create_computational_graph
from activations import Sigmoid

def main():
    a = Tensor(-2.0)
    b = Tensor(5.0)
    c = Tensor(2)
    d = a * b
    e = c + d
    f = e / c
    g = f - (d ** 2)
    g = Sigmoid()(g)
    
    print(g)
    
    create_computational_graph(g, name='forward', render=True)
    
    g.grad = 1.0
    g.backward()
    
    create_computational_graph(g, name='backward', render=True)
    
    g.reset_grads()
    create_computational_graph(g, name='reset', render=True)
    

if __name__ == "__main__":
    main()