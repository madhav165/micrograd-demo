from graphviz import Digraph

class Graph:

    def __init__(self, root):
        self.root = root

    def __repr__(self):
        return f"Graph(root={self.root}))"

    def trace(self):
        nodes, edges = set(), set()
        
        def build(v):
            if v not in nodes:
                nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
        
        build(self.root)
        return nodes, edges

    def draw_dot(self):
        pass