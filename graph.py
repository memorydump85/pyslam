import numpy as np
from collections import namedtuple

from mmath import *

import pyximport; pyximport.install()
from perfx import XYTConstraint_residual, XYTConstraint_jacobians
from perfx import cycle_2PI_towards_zero



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

    `xyt` is the rigidbody transformation `T`, between the vertices
    `v_out` and `v_in`, expressed using its parameters of x:
    displacement, y: displacement and theta: rotation

    This constraint constrains the `xyt` between `v_out` and `v_in` to
    be distributed according to a specified gaussian distribution.
    """
    _DOF = 3

    def __init__(self, v_out, v_in, gaussian):
        self._vx = [ v_out, v_in ]
        self._gaussian = gaussian
        self._Sigma_ijv = _matrix_as_ijv(np.linalg.inv(gaussian.P))
        self._jacobian_ijv_cache = None

    def residual(self):
        """
        Compute the difference in transformation implied by this edge
        from `self._gaussian.mu`.
        """
        stateA, stateB = self._vx[0].state, self._vx[1].state
        return XYTConstraint_residual(self._gaussian.mu, stateA, stateB)

    def chi2(self):
        z = self.residual()
        return reduce(np.dot, [ z.T, self._gaussian.P, z ])

    def uncertainty(self, roff=0, coff=0):
        off = np.array([ roff, coff, 0.]).reshape((3, 1))
        return self._Sigma_ijv + off

    def jacobian(self, roff=0):
        """
        Compute the jacobian matrix of the residual error function
        evaluated at the current states of the connected vertices.

        returns the sparse Jacobian matrix entries in triplet format
        (i,j,v). The row index of the entries is offset by `roff`.

        It is useful to specify `roff` when this Jacobian matrix is
        computed as a sub-matrix of the graph Jacobian.
        """
        Ja, Jb = XYTConstraint_jacobians(self._vx[0].state, self._vx[1].state)

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
    Anchors the `xyt` parameters of a vertex `v` to conform to a
    gaussian distribution. The most common use of this edge type is to
    anchor the `xyt` of the first node in a SLAM graph to a fixed value.
    This prevents the graph solution from drifting arbitrarily.
    """
    _DOF = 3

    def __init__(self, v, gaussian):
        self._vx = [v]
        self._gaussian = gaussian
        self._Sigma_ijv = _matrix_as_ijv(np.linalg.inv(gaussian.P))
        self._jacobian_ijv_cache = None

    def residual(self, aggregate_state=None):
        r = self._gaussian.mu - self._vx[0].state
        r[2] = cycle_2PI_towards_zero(r[2])
        return r

    def chi2(self):
        return self._gaussian.chi2(self._vx[0].state)

    def uncertainty(self, roff=0, coff=0):
        off = np.array([ roff, coff, 0.]).reshape((3, 1))
        return self._Sigma_ijv + off

    def jacobian(self, roff=0, eps=1e-5):
        """
        Compute the jacobian matrix of the residual error function
        evaluated at the current states of the connected vertices.

        Returns a (dok format) sparse matrix since the jacobian of an
        edge constraint is sparse. The `graph_state_length` parameter is
        required to fix the column dimension of this sparse matrix.
        Thus, the sparse matrix has `graph_state_length` columns and
        `len(self.residual())` rows.
        """
        if self._jacobian_ijv_cache is None:
            J = -np.eye(3)
            ndx = self._vx[0]._graph_state_ndx
            self._jacobian_ijv_cache = _matrix_as_ijv(J, i0=roff, j0=ndx)

        return self._jacobian_ijv_cache


GraphStats = namedtuple('GraphStats',
                ['chi2', 'chi2_N', 'DOF'])


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

    def get_stats(self):
        DOF = sum(e._DOF for e in self.edges) - len(self.state)
        chi2 = sum(e.chi2() for e in self.edges)
        return GraphStats(chi2, chi2/DOF, DOF)
