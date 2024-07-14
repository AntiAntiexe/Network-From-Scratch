import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2*x**2


x = np.arange(0, 5,0.001)
y = f(x)

plt.plot(x, y)

p2_delta = 0.0001
x1 = 2
x2 = x1 + p2_delta

y1 = f(x1)
y2 = f(x2)

print((x1,y1),(x2, y2))

approx_deriv = (y2-y1)/(x2-x1)

b = y2  - approx_deriv*x2


def tangent_line(x):
    return approx_deriv*x + b


to_plot = [x1-0.9, x1, x1+0.9]

plt.plot(to_plot, [tangent_line(x) for x in to_plot])

print('Approximate Derivative for f(x)',
      f'where x = {x1} is {approx_deriv}')


plt.show()
