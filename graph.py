import numpy as np
from math import sin, cos
from scipy.sparse import dok_matrix

from mmath import *

import pyximport; pyximport.install()
from perfx import XYTConstraint_residual, XYTConstraint_jacobians



#--------------------------------------
class VertexXYT(object):
#--------------------------------------
    def __init__(self, xyt0):
        self.state0 = xyt0


def _matrix_as_ijv(M, i0=0, j0=0):
    i, j = np.indices(M.shape).reshape(2, -1)
    v = M.ravel()
    return np.array([ i+i0, j+j0, v ])


#--------------------------------------
class XYTConstraint(object):
#--------------------------------------
    """
    Constrain the transformation between two `VertexXYT`s.

    `xyt` is the rigidbody transformation `T`, between the vertices `v_out` and
    `v_in`, expressed using its parameters of x: displacement, y: displacement
    and theta: rotation

    This constraint constrains the `xyt` between `v_out` and `v_in` to be
    distributed according to a specified gaussian distribution.
    """
    def __init__(self, v_out, v_in, gaussian):
        self._vx = [ v_out, v_in ]
        self._gaussian = gaussian
        self._Sigma_ijv = _matrix_as_ijv(np.linalg.inv(gaussian.P))
        self._jacobian_ijv_cache = None

    def aggregate_vertex_state(self):
        """
        Stacked state of all connected vertices
        """
        return np.concatenate([ v.state for v in self._vx ])

    def residual(self, aggregate_state=None):
        """
        Compute the difference in transformation from `self._gaussian.mu` given
        the `aggregate_state` of vertices. If `aggregate_state` is not
        specified, the current aggregate state from
        `self.aggregate_vertex_state()` is used.
        """
        if aggregate_state is None:
            stateA, stateB = self._vx[0].state, self._vx[1].state
        else:
            stateA, stateB = aggregate_state[0:3], aggregate_state[3:6]

        return XYTConstraint_residual(self._gaussian.mu, stateA, stateB)

    def chi2(self):
        current_xyt = xyt_inv_mult(self._vx[0].state, self._vx[1].state)
        return self._gaussian.chi2(current_xyt)

    def uncertainty(self, roff=0, coff=0):
        off = np.array([ roff, coff, 0.]).reshape((3, 1))
        return self._Sigma_ijv + off

    def jacobian(self, roff=0, eps=1e-5,):
        """
        Compute the jacobian matrix of the residual error function evaluated at
        the current states of the connected vertices.

        Returns a (dok format) sparse matrix since the jacobian of an edge
        constraint is sparse. The `graph_state_length` parameter is required to
        fix the column dimension of this sparse matrix. Thus, the sparse matrix
        has `graph_state_length` columns and `len(self.residual())` rows.
        `
        """
        # xa, ya, ta = self._vx[0].state
        # xb, yb, tb = self._vx[1].state
        # sa, ca = sin(ta), cos(ta)

        # Ja = [ ca,  sa,  sa*(xb-xa)-ca*(yb-ya),
        #       -sa,  ca,  ca*(xb-xa)+sa*(yb-ya),
        #        0.,   0.,                    1. ]

        # Jb = [ -ca,  -sa,  0.,
        #         sa,  -ca,  0.,
        #         0.,   0., -1. ]

        Ja, Jb = XYTConstraint_jacobians(self._vx[0].state, self._vx[1].state)

        # J = np.vstack(( Ja, Jb ))
        # J = np.array([
        #         [ ca,  sa,  sa*(xb-xa)-ca*(yb-ya),  -ca,  -sa,  0.],
        #         [-sa,  ca,  ca*(xb-xa)+sa*(yb-ya),   sa,  -ca,  0.],
        #         [ 0.,   0.,                    1.,   0.,   0., -1.]])

        # Above is the analytical version of:
        #   J = numerical_jacobian(self.residual, x0=self.aggregate_vertex_state(), eps=eps)

        if self._jacobian_ijv_cache is None:
            ndx0 = self._vx[0]._graph_state_ndx
            ndx1 = self._vx[1]._graph_state_ndx
            self._jacobian_ijv_cache = np.concatenate([
                            _matrix_as_ijv(np.array(Ja).reshape(3,3), i0=roff, j0=ndx0),
                            _matrix_as_ijv(np.array(Jb).reshape(3,3), i0=roff, j0=ndx1)
                        ], axis=1)

        self._jacobian_ijv_cache[2] = Ja + Jb
        return self._jacobian_ijv_cache


#--------------------------------------
class AnchorConstraint(object):
#--------------------------------------
    """
    Anchors the `xyt` parameters of a vertex `v` to conform to a gaussian
    distribution. The most common use of this edge type is to anchor the `xyt`
    of the first node in a SLAM graph to a fixed value. This prevents the graph
    solution from drifting arbitrarily.
    """
    def __init__(self, v, gaussian):
        self._vx = v
        self._gaussian = gaussian
        self._Sigma_ijv = _matrix_as_ijv(np.linalg.inv(gaussian.P))
        self._jacobian_ijv_cache = None

    def aggregate_vertex_state(self):
        """
        Stacked state of all connected vertices. This type of edge has only one
        vertex.
        """
        return self._vx.state

    def residual(self, aggregate_state=None):
        if aggregate_state is None:
            aggregate_state = self.aggregate_vertex_state()

        r = self._gaussian.mu - aggregate_state
        r[2] = rotate_angle_towards_zero(r[2])
        return r

    def chi2(self):
        return self._gaussian.chi2(self._vx.state)

    def uncertainty(self, roff=0, coff=0):
        off = np.array([ roff, coff, 0.]).reshape((3, 1))
        return self._Sigma_ijv + off

    def jacobian(self, roff=0, eps=1e-5):
        """
        Compute the jacobian matrix of the residual error function evaluated at
        the current states of the connected vertices.

        Returns a (dok format) sparse matrix since the jacobian of an edge
        constraint is sparse. The `graph_state_length` parameter is required to
        fix the column dimension of this sparse matrix. Thus, the sparse matrix
        has `graph_state_length` columns and `len(self.residual())` rows.
        `
        """
        if self._jacobian_ijv_cache is None:
            J = -np.eye(3)
            # Above is the analytical version of:
            #   J = numerical_jacobian(self.residual, x0=self.aggregate_vertex_state(), eps=eps)

            ndx = self._vx._graph_state_ndx
            self._jacobian_ijv_cache = _matrix_as_ijv(J, i0=roff, j0=ndx)

        return self._jacobian_ijv_cache


#--------------------------------------
class Graph(object):
#--------------------------------------
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.state = np.concatenate([ v.state0 for v in self.vertices ])

        # The vertices get views into the graph's state
        slice_end = np.cumsum([ len(v.state0) for v in self.vertices ]).tolist()
        slice_start = [0] + slice_end[:-1]
        for v, i, j in zip(self.vertices, slice_start, slice_end):
            v.state = self.state[i: j]
            v._graph_state_ndx = i

    def anchor_first_vertex(self):
        v0 = self.vertices[0]

        mu = v0.state.copy()
        P = 1000. * np.eye(len(mu))
        self._anchor = AnchorConstraint(v0, MultiVariateGaussian(mu, P))

        self.edges.append(self._anchor)
