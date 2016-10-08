import numpy as np



def xyt_inv_mult(a, b):
    """
    compute `A.inv() * B` for two rigid body transforms
    `A` and `B`.

    Parameters
    ----------
        `a`: Rigid body transform `A` expressed as an xyt array
        `b`: Rigid body transform `B` expressed as an xyt array

    Returns
    -------
        an xyt array representing the result of `A.inv() * B`
    """
    theta = a[2]
    ca, sa = np.cos(theta), np.sin(theta)
    dx, dy, dt = b - a
    return np.array([ ca*dx + sa*dy, -sa*dx + ca*dy, dt ])


def xyt_mult(a, b):
    """
    compute `A * B` for two rigid body transforms `A` and `B`.

    Parameters
    ----------
        `a`: Rigid body transform `A` expressed as an xyt array
        `b`: Rigid body transform `B` expressed as an xyt array

    Returns
    -------
        an xyt array representing the result of `A * B`
    """
    sa, ca = np.sin(a[2]), np.cos(a[2]);

    return np.array([ ca*b[0] - sa*b[1] + a[0],
                      sa*b[0] + ca*b[1] + a[1],
                      a[2] + b[2] ])


def numerical_jacobian(f, x0, eps=1e-5):
    """
    Compute jacobian via numerical differentiation. This implementation
    uses forward differences.

    Parameters
    ----------
        `f`: is a vector-valued function.
        `x0`: The point at which numerical differentiation is performed

    Return value
    ------------
        Returns an `M x N` matrix where `M` = `len(f(x0))` `N` =
        `len(x0)`
    """
    f_x0 = f(x0)
    I_eps = np.eye(len(x0)) * eps

    def one_hot_delta(pos):
        """ vector with just one position set to eps """
        return I_eps[pos]

    def partial_derivative(nth):
        d = one_hot_delta(nth)
        return (f(x0+d) - f_x0) / eps

    # variables change along columns. function outputs along rows
    #           _                                  _
    #          |    df_1      df_1           df_1   |
    #          |   ------    ------   ...   ------  |
    #          |    dx_1      dx_2           dx_n   |
    #          |                                    |
    #     J =  |    df_2       .               .    |
    #          |   ------          .           .    |
    #          |    dx_1               .       .    |
    #          |                           .   .    |
    #          |    df_m                     df_m   |
    #          |   ------     ...     ...   ------  |
    #          |    dx_1                     dx_n   |
    #          |_                                  _|
    #
    return np.array([ partial_derivative(i) for i in xrange(len(x0)) ]).T


#--------------------------------------
class MultiVariateGaussian(object):
#--------------------------------------
    """
    MultivariateGaussian specified with its mean `mu` and
    precision/information matrix `P`. Alternatively, the precision
    matrix `P` is the inverse of the covariance matrix.
    """
    def __init__(self, mu, P):
        self.mu = mu
        self.P = P

    def chi2(self, x):
        z = x - self.mu
        return reduce(np.dot, [ z.T, self.P, z ])
