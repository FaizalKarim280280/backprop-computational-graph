from tensor import Tensor
from plot_graph import create_computational_graph

def main():
    a = Tensor(-2)
    b = Tensor(5)
    c = Tensor(10)
    d = a * b
    e = c + d
    f = e / c
    g = f - d ** 2

    print(g)
    
    create_computational_graph(g, name='forward', render=True)
    
    g.grad = 1.0
    g.backward()
    
    create_computational_graph(g, name='backward', render=True)

if __name__ == "__main__":
    main()