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

o.grad = 1.0
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

# # Manual
# # do/do = 1
# o.grad = 1.0
# # do/dn = 1 - o**2
# n.grad = 1 - o.data**2
# # do/dx1w1x2w2 = do/dn * dn/dx1w1x2w2
# x1w1x2w2.grad = n.grad * 1.0
# # do/db = do/dn * dn/db
# b.grad = n.grad * 1.0
# # do/dx1w1 = do/dx1w1x2w2 * dx1w1x2w2/dx1w1
# x1w1.grad = x1w1x2w2.grad * 1.0
# # do/dx2w2 = do/dx1w1x2w2 * dx1w1x2w2/dx2w2
# x2w2.grad = x1w1x2w2.grad * 1.0
# # do/x1 = do/dx1w1 * dx1w1/dx1
# x1.grad = x1w1.grad * w1.data
# # do/w1 = do/dx1w1 * dx1w1/dw1
# w1.grad = x1w1.grad * x1.data
# # do/x2 = do/dx2w2 * dx2w2/dx2
# x2.grad = x2w2.grad * w2.data
# # do/w2 = do/dx2w2 * dx2w2/dw2
# w2.grad = x2w2.grad * x2.data

