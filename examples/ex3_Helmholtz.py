import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from chebyspectral import *
import time

deg = 24  # degree of Chebyshev polynomial

# Source term
f = lambda xq, yq: np.exp(-10*((yq-1)**2 + (xq-0.5)**2))
fHat = chebfit2d(f, deg)

# Boundary conditions
bc_value = lambda xq: 0
bc_derivative_order = 0  # Dirichlet, 0-th derivative
bc_position = -1
bc_axis = 0 # x-axis
bc_value_hat = chebfit(bc_value, deg)

bc_x_1 = [bc_value_hat, bc_derivative_order, bc_position, bc_axis]
bc_x_2 = [bc_value_hat, bc_derivative_order, -bc_position, bc_axis]
bc_y_1 = [bc_value_hat, bc_derivative_order, bc_position, 1]
bc_y_2 = [bc_value_hat, bc_derivative_order, -bc_position, 1]

bc = [bc_x_1, bc_x_2, bc_y_1, bc_y_2]

# Differentiation matrix
l_operator = [[9**2, 0, 1], [0, 0, 1]]
L = chebdiff(l_operator, deg)
L, fHat = chebbc(L, fHat, bc)

# Compute solution
t0 = time.time()
uHat = np.dot(np.linalg.pinv(L), fHat)
uHat = uHat.reshape((deg+1, deg+1))
print('Elapsed time: {}s'.format(np.round(time.time()-t0,2)))

u_analytical = 0.01172257000
u_sol = chebeval(uHat, 0, 0)
print('Absolute error at (0,0): {}'.format(np.abs(u_analytical - u_sol)))

# Plot solution
N = 100
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
xx, yy = np.meshgrid(x, y)

u = chebeval(uHat, x, y)
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, u, cmap=cm.viridis,
                linewidth=0, antialiased=False)
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('f(x,y)')

fig = plt.figure(2)
plt.contourf(xx, yy, u, 100, cmap=cm.jet)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()