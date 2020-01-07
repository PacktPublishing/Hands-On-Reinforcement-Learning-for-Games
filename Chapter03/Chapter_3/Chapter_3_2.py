import matplotlib.pyplot as plt
from random import random

ins = 0
n = 1000

x_ins = []
y_ins = []
x_outs = []
y_outs = []

for _ in range(n):
    x = (random()-.5) * 2
    y = (random()-.5) * 2 
    if (x**2+y**2) <= 1:
        ins += 1
        x_ins.append(x)
        y_ins.append(y)
    else:
        x_outs.append(x)
        y_outs.append(y)

pi = 4 * ins/n
print(pi)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.scatter(x_ins, y_ins, color='g', marker='s')
ax.scatter(x_outs, y_outs, color='r', marker='s')
plt.show()

