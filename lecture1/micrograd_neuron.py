import math
import numpy as np
import matplotlib.pyplot as plt
import logging
from graph import Graph
from value import Value

logging.getLogger().setLevel(logging.INFO)

# change to directory of the file
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# inputs x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# weights w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# bias
b = Value(6.8813735870195432, label='b')

# neuron model
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label='n'
o = n.tanh(); o.label='o'

# set to intialize _backward() correctly at output
o.grad = 1.0

# calculating the grad one node at a time in reverse order
o._backward()
n._backward()
x1w1x2w2._backward()
b._backward()
x1w1._backward()
x2w2._backward()
x1._backward()
w1._backward()
x2._backward()
w2._backward()

g = Graph(o)
gout = g.draw_dot()
gout.render('gout_neuron')

topo = g.build_topo()
logging.info(topo)