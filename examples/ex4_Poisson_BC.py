import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from chebyspectral import *
import time

deg = 24  # degree of Chebyshev polynomial

# Source term
f = lambda xq, yq: 0*xq*yq
fHat = chebfit2d(f, deg)

# Boundary conditions
f_bc_x_a = lambda xq: 0
f_bc_y_a = lambda xq: 0
f_bc_x_b = lambda xq: 1/5.0*np.sin(3*np.pi*xq)
f_bc_y_b = lambda xq: np.sin(np.pi*xq)**4 * (xq<0)

# Boundary conditions
bc_derivative_order = 0  # Dirichlet, 0-th derivative
bc_position = -1
bc_axis = 0 # x-axis

bc_x_1 = [chebfit(f_bc_x_a, deg), bc_derivative_order, bc_position, bc_axis]
bc_x_2 = [chebfit(f_bc_x_b, deg), bc_derivative_order, -bc_position, bc_axis]
bc_y_1 = [chebfit(f_bc_y_a, deg), bc_derivative_order, bc_position, 1]
bc_y_2 = [chebfit(f_bc_y_b, deg), bc_derivative_order, -bc_position, 1]

bc = [bc_x_1, bc_x_2, bc_y_1, bc_y_2]

# Differentiation matrix
l_operator = [[0, 0, 1], [0, 0, 1]]
L = chebdiff(l_operator, deg)
L, fHat = chebbc(L, fHat, bc)

# Compute solution
t0 = time.time()
uHat = np.dot(np.linalg.pinv(L), fHat)
uHat = uHat.reshape((deg+1, deg+1))
print('Elapsed time: {}s'.format(np.round(time.time()-t0,2)))

u_analytical = 0.0495946503
u_sol = chebeval(uHat, 0, 0)
print('Absolute error at (0, 0): {}'.format(np.abs(u_analytical - u_sol)))

# Plot solution
N = 100
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
xx, yy = np.meshgrid(x, y)

u = chebeval(uHat, x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, u, cmap=cm.viridis,
                linewidth=0, antialiased=False)
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('f(x,y)')
plt.show()
