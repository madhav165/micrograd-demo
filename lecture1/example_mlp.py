from mlp import MLP
from graph import Graph
import logging

logging.getLogger().setLevel(logging.INFO)
# log = logging.getLogger(__name__)
# log.setLevel(logging.INFO)


x = [2.0, 3.0, -1.0]
m = MLP(3, [4, 4, 1])
o = m(x)
logging.info(f'o = {o}')
g = Graph(o)
gout = g.draw_dot()
gout.render('gout_mlp')

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
    ]
ys = [1.0, -1.0, -1.0, 1.0]

ypred = [m(x) for x in xs]
logging.info(f'ypred = {ypred}')

loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
logging.info(f'loss = {loss}')
