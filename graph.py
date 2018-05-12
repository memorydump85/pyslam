from six.moves import reduce

from collections import namedtuple
import numpy as np
import scipy.sparse as sp

from mmath import xyt_inv_mult, xyt_mult, MultiVariateGaussian

import pyximport; pyximport.install()
from perfx import XYTConstraint_residual, XYTConstraint_jacobians
from perfx import normalize_angle_range



#
# Some specializations for performance
#

def _hstack2d(arr_collection):
    """ faster than `np.hstack` because it avoids calling `np.atleast1d` """
    return np.concatenate(arr_collection, axis=1)


def _hstack1d(arr_collection):
    """ faster than `np.hstack` because it avoids calling `np.atleast1d` """
    return np.concatenate(arr_collection, axis=0)


class coo_distinct_matrix(sp.coo_matrix):
    """ A COO sparse matrix that assumes that there are no duplicate entries """

    def sum_duplicates(self):
        pass


#--------------------------------------
class VertexXYT(object):
#--------------------------------------
    def __init__(self, xyt0):
        self.state0 = xyt0


#--------------------------------------
class _MatrixDataIJV(object):
#--------------------------------------
    """ Encapsulation for matrix (i,j,v) data stored in a contiguous
    record array.
    """
    __slots__ = [ '_data' ]

    def __init__(self, ijv_records):
        self._data = np.array(ijv_records, [ ('i', '<i4'), ('j', '<i4'), ('v', '<f8') ])

    @classmethod
    def from_ijv(cls, i, j, v):
        records = [ (ii, jj, vv) for ii, jj, vv in zip(i, j, v) ]
        return cls(records)

    @classmethod
    def from_dense(cls, M, i_offset=0, j_offset=0):
        i, j = np.indices(M.shape).reshape(2, -1) + np.c_[[i_offset, j_offset]]
        return cls.from_ijv(i, j, M.ravel())

    def indices(self):
        return self._data.view(dtype=np.int32).reshape((-1, 4))[:,:2].T

    @property
    def i(self):
        return self.indices()[0]

    @property
    def j(self):
        return self.indices()[1]

    @property
    def v(self):
        return self._data.view(dtype=np.float64).reshape((-1, 2))[:,1].T

    @v.setter
    def v(self, value):
        self._data.view(dtype=np.float64).reshape((-1, 2))[:,1] = value

    def offset(self, i, j):
        x = self._data.copy()
        x.view(dtype=np.int32).reshape((-1, 4))[:,:2] += np.array([i, j], dtype=np.int32)
        return self.__class__(x)


def _stack_ijv(ijv_data_list):
    return _MatrixDataIJV(np.concatenate([ x._data for x in ijv_data_list ]))


#--------------------------------------
class XYTConstraint(object):
#--------------------------------------
    """
    Constrain the transformation between two `VertexXYT`s.

    `xyt` is the rigidbody transformation `T`, between the vertices
    `v_out` and `v_in`, expressed using its parameters of x:
    displacement, y: displacement and theta: rotation

    This constraint constrains the `xyt` between `v_out` and `v_in` to
    be distributed according to the specified gaussian distribution.
    """
    _DOF = 3

    def __init__(self, v_out, v_in, gaussian):
        self._vx = [ v_out, v_in ]
        self._gaussian = gaussian
        self._Sigma_ijv = _MatrixDataIJV.from_dense(np.linalg.inv(gaussian.P))
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
        return self._Sigma_ijv.offset(roff, coff)

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
            self._jacobian_ijv_cache = _stack_ijv([
                            _MatrixDataIJV.from_dense(np.array(Ja).reshape(3,3), roff, ndx0),
                            _MatrixDataIJV.from_dense(np.array(Jb).reshape(3,3), roff, ndx1)
                        ])

        self._jacobian_ijv_cache.v = Ja + Jb
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
        self._Sigma_ijv = _MatrixDataIJV.from_dense(np.linalg.inv(gaussian.P))
        self._jacobian_ijv_cache = None

    def residual(self, aggregate_state=None):
        r = self._gaussian.mu - self._vx[0].state
        r[2] = normalize_angle_range(r[2])
        return r

    def chi2(self):
        return self._gaussian.chi2(self._vx[0].state)

    def uncertainty(self, roff=0, coff=0):
        return self._Sigma_ijv.offset(roff, coff)

    def jacobian(self, roff=0, eps=1e-5):
        """
        Compute the jacobian matrix of the residual error function
        evaluated at the current states of the connected vertices.

        returns the sparse Jacobian matrix entries in triplet format
        (i,j,v). The row index of the entries is offset by `roff`.

        It is useful to specify `roff` when this Jacobian matrix is
        computed as a sub-matrix of the graph Jacobian.
        """
        if self._jacobian_ijv_cache is None:
            J = -np.eye(3)
            ndx = self._vx[0]._graph_state_ndx
            self._jacobian_ijv_cache = _MatrixDataIJV.from_dense(J, roff, ndx)

        return self._jacobian_ijv_cache


GraphStats = namedtuple('GraphStats', ['chi2', 'chi2_N', 'DOF'])


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
        if hasattr(self, '_anchor') == False:
            v0 = self.vertices[0]

            mu = v0.state.copy()
            P = 1000. * np.eye(len(mu))
            self._anchor = AnchorConstraint(v0, MultiVariateGaussian(mu, P))

            self.edges.append(self._anchor)

    def get_stats(self):
        original_edges = [ e for e in self.edges if e is not self._anchor ]

        DOF = sum(e._DOF for e in original_edges) - len(self.state)
        chi2 = sum(e.chi2() for e in original_edges)

        return GraphStats(chi2, chi2/DOF, DOF)

    def get_linearization(self):
        """
        Linearizes the non-linear constraints in this graph, at its
        current state `self.state`, to produce an approximating linear
        system.

        Returns:
        --------
            `W`:
                Weighting matrix for the linear constraints
            `J`:
                Jacobian of the system at `self.state`.
            `r`:
                vector of stacked residuals

        The linear system `W J x = W r` captures an approximate linear
        model of the graph constraints that is valid near the current
        state of the graph.
        """
        edges = self.edges
        residuals = [ e.residual() for e in edges ]

        # For each edge jacobian, compute the its row index in the
        # graph jacobian `J`
        residual_lengths = [ len(r) for r in residuals ]
        row_offsets = [0,] + np.cumsum(residual_lengths).tolist()

        # Stack edge jacobians to produce system jacobian
        j_coo = _stack_ijv([ e.jacobian(roff=r) for r, e in zip(row_offsets, edges) ])
        J = coo_distinct_matrix((j_coo.v, (j_coo.i, j_coo.j)))

        # Stack edge weights to produce system weights
        w_coo = _stack_ijv([ e.uncertainty(r, r) for r, e in zip(row_offsets, edges) ])
        W = coo_distinct_matrix((w_coo.v, (w_coo.i, w_coo.j))).tocsc()

        r = _hstack1d(residuals)

        return W, J, r
