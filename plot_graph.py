from graphviz import Digraph
import os

def get_graph(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.operand:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def create_computational_graph(root, name, render=False):
    nodes, edges = get_graph(root)
    
    graph = Digraph(format='png', graph_attr={'rankdir': 'LR', 'label': f'Computation Graph [Gradients {"not " if not root.gradients_calculated else ""}calculated]', 'labelloc' : 't'})
    
    for n in nodes:
        node_name = f'{n.name} |' if n.name is not None else ""
        graph.node(name=str(id(n)), 
                   label = f" {node_name} value={n.value:.5f} | grad={n.grad:.5f}", 
                   shape='record', 
                   style='filled', 
                   fillcolor='lightgreen' if n.leaf else 'lightpink')
        
        if n.operation is not None:
            graph.node(name=str(id(n)) + n.operation, label=n.operation, color='darkblue', style='filled', fillcolor='cyan')
            graph.edge(str(id(n)) + n.operation, str(id(n)))
    
    for n1, n2 in edges:
        graph.edge(str(id(n1)), str(id(n2)) + n2.operation)
        
    if render:
        graph.render(os.path.join('plots', name), cleanup=True)
    else:
        return graph

