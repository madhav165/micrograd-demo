import math
import numpy as np
import matplotlib.pyplot as plt
import logging
from graph import Graph

logging.getLogger().setLevel(logging.INFO)

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
logging.info(f"a = {a}")
logging.info(f"b = {b}")
logging.info(f"c = {c}")
e = a * b
e.label = 'e'
d = e + c
d.label = 'd'
logging.info(f"d = a * b + c = {d}")
logging.info(f'd._prev = {d._prev}')
logging.info(f'd._op = {d._op}')

g = Graph(d)
logging.info(f'g = {g}')
logging.info(f'g.trace = {g.trace()}')