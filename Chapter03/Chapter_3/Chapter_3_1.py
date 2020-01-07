from random import *
from math import sqrt

ins = 0
n = 100000

for i in range(0, n):
    x = (random()-.5) * 2
    y = (random()-.5) * 2
    if sqrt(x*x+y*y)<=1:
        ins+=1

pi = 4 * ins / n
print(pi)
