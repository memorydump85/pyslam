import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport sin, cos, fmod, M_PI



cpdef inline double normalize_angle_range(double t):
    """ Normalize angle to be in [-PI, +PI] """
    if t > 0:
        return fmod(t+M_PI, 2.0*M_PI) - M_PI
    else:
        return fmod(t-M_PI, 2.0*M_PI) + M_PI


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

    r[2] = normalize_angle_range(r[2])

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
