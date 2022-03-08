"""
====================================================
Chebyshev Spectral Methods Module
====================================================

This module provides a number of functions useful for computing approximate solutions
to linear (ordinary and partial) differential equations on a square domain.

The functions use the Tau method to compute solutions and the boundary conditions
(Dirichlet or Neumann) can be customized individually on all boundary domains.
The methods are limited to linear differential operators with constant coefficients.

"""
import numpy as np

__all__ = [
    'chebfit', 'chebfit2d', 'chebeval', 'chebmat', 'chebdiff', 'chebbc']


def chebfit(fun, deg):
    """
    Find the 1D coefficients of the Chebyshev series for the provided function

    This function uses a Chebyshev-Gauss-Lobatto quadrature to compute the
    coefficients.

    Parameters
    ----------
    fun : function
        function to be approximated. Takes one variable argument.
    deg : int
        Degree of the polynomial approximation

    Returns
    -------
    f_hat : ndarray
        1D array containing the approximated coefficients of the Chebyshev series
        to the requested degree

    """
    x = np.array([-np.cos(np.pi * i / deg) for i in range(deg + 1)])
    f_val = fun(x)
    w = np.array((deg-1)*[np.pi/deg])
    w = np.append(w, np.pi/(2*deg))
    w = np.insert(w, 0, np.pi/(2*deg))

    cheby_mat = chebmat(deg)
    poly_points = np.array([x**i for i in range(len(x))])

    poly = np.dot(cheby_mat, poly_points)
    gamma_n = np.dot(poly**2, w)
    f_hat = 1.0/gamma_n*np.dot(poly, f_val*w)
    return f_hat


# Compute polynomial coefficients of 2D function using Chebyshev-Gauss-Lobatto quadrature
def chebfit2d(fun, deg):
    """
    Find the 2D coefficients of the Chebyshev series for the provided function

    This function uses a Chebyshev-Gauss-Lobatto quadrature to compute the
    coefficients.

    Parameters
    ----------
    fun : function
        function to be approximated. Takes one variable argument.
    deg : int
        Degree of the polynomial approximation

    Returns
    -------
    f_hat : ndarray
        2D array containing the approximated coefficients of the Chebyshev series
        to the requested degree

    """
    x = np.array([-np.cos(np.pi * i / deg) for i in range(deg + 1)])
    f = fun(x.reshape(deg+1,1), x.reshape(1,deg+1)).T
    w = np.array((deg-1)*[np.pi/deg])
    w = np.append(w, np.pi/(2*deg))
    w = np.insert(w, 0, np.pi/(2*deg))
    w = w.reshape(len(w), 1)

    cheby_mat = chebmat(deg)
    poly_points = np.array([x**i for i in range(len(x))])

    poly = np.dot(cheby_mat, poly_points)
    gamma_n = np.linalg.multi_dot([poly**2, w, w.T, (poly**2).T])
    f_hat = (1.0 / gamma_n) *  np.linalg.multi_dot([poly * w.T, f, (poly * w.T).T])
    return f_hat


def chebeval(f_hat, xq, *args):
    """
    Evaluate polynomial with coefficients f_hat at query points xq and yq

    Parameters
    ----------
    f_hat : ndarray
        Polynomial coefficients. It is a flat vector for 1D evaluation and
        a matrix for 2D evaluation.
    xq : ndarray
        Query points for evaluation in the first dimension.
    args: ndarray, optional
        Extra argument to be used for 2D evaluation, in which case it should
        contain the query points in the second dimension.

    Returns
    -------
    f : ndarray
        1D or 2D array containing the evaluation on the query points

    """
    deg = f_hat.shape[0]-1
    cheby_mat = chebmat(deg)

    poly_points_x = np.array([xq ** i for i in range(deg + 1)])
    poly_x = np.dot(cheby_mat, poly_points_x)
    if args:
        yq = args[0]
        poly_points_y = np.array([yq ** i for i in range(deg + 1)])
        poly_y = np.dot(cheby_mat, poly_points_y)
        f = np.linalg.multi_dot([poly_x.T, f_hat, poly_y])
    else:
        f = np.linalg.multi_dot([poly_x.T, f_hat])

    return f


def chebmat(degree):
    """
    Polynomial coefficient matrix.

    Returns matrix that when multiplied with column vector [1, x, x^2 ,...,x^degree]
    yields the Chebyshev basis up to 'degree'

    Parameters
    ----------
    degree : int
        Degree of polynomial

    Returns
    -------
    a : ndarray
        2D array containing the polynomial coefficient matrix

    """
    a = np.zeros((degree+1, degree+1))
    a[0, 0] = 1
    a[1, 1] = 1
    for i in range(2, degree+1):
        a_slice = a[i-1, 0:-1]
        a[i, :] = np.insert(2*a_slice, 0, 0)-a[i-2, :]
    return a


def chebdiff(l_operator, degree):
    """
    Spectral differentiation matrix.

    Returns the spectral differentiation matrix for the linear differential
    operator l_operator.

    Parameters
    ----------
    l_operator : array_like
        If the problem is 1D, it contains a list with the the coefficients of the
        derivatives, l_operator = [0-th order coefficient, 1-st order coefficient, ...]
        Example: L = d^3/dx^3 - 2 d/dx + 3 --> l_operator = [3, -2, 0, 1]
        If the problem is 2D, it contains a nested list where the first sublist contains
        the coefficient in the first dimension, and the second sublist the coefficients
        in the second dimension.
        Example: L = d^2/dx^2 - 2*d^2/dy^2 --> l_operator = [[0, 0, 1],[0, 0, -2]]

    degree : int
        Degree of polynomial approximation

    Returns
    -------
    l_mat : ndarray
        2D array containing the spectral differentiation matrix. Contains
        a matrix in the 1D case, and a list of matrices in the 2D case.

    """
    cheby_mat = chebmat(degree)
    x_diff = np.diag(np.linspace(1, degree, degree), -1)
    l_first_order = np.round(np.linalg.multi_dot([cheby_mat, x_diff, np.linalg.pinv(cheby_mat)]))

    if isinstance(l_operator[0], list):
        dim = 2
        l_current = l_operator[0]
        l_mat = [[0], [0]]
    else:
        dim = 1
        l_current = l_operator
        l_mat = np.zeros((degree + 1, degree + 1))

    for k in range(dim):
        l_tmp = np.zeros((degree + 1, degree + 1))
        for i in range(len(l_current)):
            if i == 0:
                l_tmp += l_current[i]*np.eye(degree+1)
            elif i == 1:
                l_tmp += l_current[i] * l_first_order
            else:
                chain = i*[l_first_order]
                l_tmp += l_current[i]*np.linalg.multi_dot(chain)

        l_tmp = l_tmp.T
        if dim == 2:
            l_mat[k] = l_tmp
            l_current = l_operator[1]
        else:
            l_mat = l_tmp

    return l_mat


def chebbc(l_mat, rhs, bc):
    """
    Apply boundary conditions.

    Applies the provided boundary conditions to the differentiation matrix and
    the right hand side.

    Parameters
    ----------
    l_mat : ndarray
        The spectral differentiation matrix as computed by chebdiff. Contains
        a matrix in the 1D case, and a list of matrices in the 2D case.

    rhs : ndarray
        polynomial coefficients of right hand side / source term of
        the differential equation. It is a flat vector in the 1D case
        and a matrix in the 2D case.

    bc : array_like
        Boundary condition information. This should have the structure
        bc = [bc_1, bc_2, ..., bc_N], if N boundary conditions are to be applied, where
        bc_i = [value, derivative order, position, independent variable axis]
        with 'value' containing the value at the boundary (in the 1D case), or the
        coefficients of the Chebyshev series of the boundary condition (in the 2D case),
        'derivative order' defining whether it is a Dirichlet boundary condition
        (=0) or a Neumann boundary condition (>0), 'position' defining whether the
        left (=-1) or right (=1) boundary is addressed, and 'independent variable axis'
        defining whether the boundary condition applies to the x-axis (=0) or y-axis (=1).

    Returns
    -------
    l_mat : ndarray
        2D array containing the spectral differentiation matrix with applied boundary
        conditions.

    rhs : ndarray
        Flat vector containing the polynomial coefficients of the right hand side of
        the differential equation with applied boundary conditions.

    """
    if isinstance(l_mat, list):
        dim = l_mat[0].shape[0]
    else:
        dim = l_mat.shape[0]
    bc = np.asarray(bc, dtype=object)

    n_bc = len(bc)
    l_bc = np.zeros((n_bc, dim))
    d = chebdiff([0, 1], dim - 1)
    for k in range(n_bc):
        if bc[k, 1]==0:
            d_bc = np.eye(dim)
        else:
            d_bc = (d.T)**bc[k, 1]

        l_bc[k,:] = np.dot(d_bc, [(bc[k, 2]) ** x for x in range(dim)])

    if rhs.ndim > 1:
        bc_x_idx = np.where(bc[:,3]==0)[0]
        bc_y_idx = np.where(bc[:,3]==1)[0]
        l_mat[0][-len(bc_x_idx):, :] = l_bc[bc_x_idx, :]
        l_mat[1][-len(bc_y_idx):, :] = l_bc[bc_y_idx, :]

        l_mat1 = np.kron(np.eye(dim), l_mat[0])
        l_mat1[-len(bc_x_idx)*dim:, :] = 0
        for i in range(1, len(bc_x_idx)+1):
            rhs[:, -i] = bc[bc_x_idx[len(bc_x_idx)-i], 0]

        l_mat2 = np.kron(l_mat[1], np.eye(dim))
        for i in range(1,len(bc_y_idx)+1):
            l_mat2[dim - i::dim, :] = 0
            rhs[-i, :] = bc[bc_y_idx[len(bc_y_idx)-i], 0]

        l_mat = l_mat1+l_mat2
        rhs = rhs.flatten()
    else:
        l_mat[-n_bc:, :] = l_bc
        rhs[-n_bc:] = bc[:, 0]

    return l_mat, rhs

