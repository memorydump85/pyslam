import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport sin, cos, fabs, M_PI



cdef inline double _cycle_2PI_towards_zero(double t):
    """
    Return the closest to zero amongst `t`, `t+2*PI`, `t-2*PI`
    """
    cdef double _2PI = 2*M_PI
    cdef double best
    best = t if (fabs(t) < fabs(t + _2PI)) else (t + _2PI)
    best = best if (fabs(best) < fabs(t - _2PI)) else (t - _2PI)
    return best


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef XYTConstraint_residual(
        np.ndarray[np.float64_t, ndim=1] observed,
        np.ndarray[np.float64_t, ndim=1] xyt_a,
        np.ndarray[np.float64_t, ndim=1] xyt_b ):

    cdef double ta, ca, sa, dx, dy, dt
    ta = xyt_a[2]
    ca = cos(ta); sa = sin(ta)
    dx = xyt_b[0] - xyt_a[0]
    dy = xyt_b[1] - xyt_a[1]
    dt = xyt_b[2] - xyt_a[2]

    cdef double ainvb[3]
    ainvb[:] = [ ca*dx + sa*dy, -sa*dx + ca*dy, dt ]

    cdef double r[3]
    r[:] = [ observed[0] - ainvb[0],
             observed[1] - ainvb[1],
             observed[2] - ainvb[2] ]

    r[2] = _cycle_2PI_towards_zero(r[2])

    return np.array(r)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef XYTConstraint_jacobians(
        np.ndarray[np.float64_t, ndim=1] xyt_a,
        np.ndarray[np.float64_t, ndim=1] xyt_b ):

    cdef double xa, ya, ta
    xa = xyt_a[0]
    ya = xyt_a[1]
    ta = xyt_a[2]

    cdef double xb, yb, tb
    xb = xyt_b[0]
    yb = xyt_b[1]
    tb = xyt_b[2]

    cdef double sa, ca
    sa = sin(ta)
    ca = cos(ta)

    cdef double Ja[9]
    Ja[:] = [ ca,  sa,  sa*(xb-xa)-ca*(yb-ya),
             -sa,  ca,  ca*(xb-xa)+sa*(yb-ya),
              0.,   0.,                    1. ]

    cdef double Jb[9]
    Jb[:] = [ -ca,  -sa,  0.,
               sa,  -ca,  0.,
               0.,   0., -1. ]

    return Ja, Jb
