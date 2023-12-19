from tensor import Tensor
from plot_graph import create_computational_graph
from activations import Sigmoid, Tanh

def main():
    a = Tensor(-2.0, name = 'a')
    b = Tensor(3.0, name = 'b')
    c = Tensor(0.5, name = 'c')
    x = Tensor(-5, name='x')
    d = a * b
    e = Tanh()(c + d)
    f = Sigmoid()(e / c)
    g = (f - (d ** 2)) * x
    g = g/a
    
    print(g)
    
    create_computational_graph(g, name='forward', render=True)
    
    g.grad = 1.0
    g.backward()
    
    create_computational_graph(g, name='backward', render=True)
    
    # g.reset_grads()
    # create_computational_graph(g, name='reset', render=True)
    

if __name__ == "__main__":
    main()