import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from chebyspectral import *

deg = 16  # degree of Chebyshev polynomial

# Source term
C = -4*np.exp(1)/(1+np.exp(2))
f = lambda xq: np.exp(xq) + C
fHat = chebfit(f, deg)

# Boundary conditions
bc_value_1 = 0
bc_derivative_order_1 = 0  # Dirichlet (0th order derivative)
bc_position_1 = -1
bc_axis_1 = 0
bc_1 = [bc_value_1, bc_derivative_order_1, bc_position_1, bc_axis_1]

bc_value_2 = 0
bc_derivative_order_2 = 0 # Dirichlet
bc_position_2 = 1
bc_axis_2 = 0
bc_2 = [bc_value_2, bc_derivative_order_2, bc_position_2, bc_axis_2]

# Differentiation matrix
l_operator = [4, -4, 1]
L = chebdiff(l_operator, deg)
L, fHat = chebbc(L, fHat, [bc_1, bc_2])

# Compute solution
u = np.dot(np.linalg.pinv(L), fHat)

# Plot solution
N = 100
x = np.linspace(-1, 1, N)
fig, ax = plt.subplots()
ax.plot(x, chebeval(u, x), 'b', label='Approximation')
ax.plot(x, np.exp(x)-np.sinh(1)/np.sinh(2)*np.exp(2*x) + C/4, 'r--', label='Exact')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
