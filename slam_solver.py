import sys
import numpy as np
import scipy.sparse as sp
from scikits.sparse.cholmod import analyze, analyze_AAt

from graphio import load_graph, render_graph_html



#
# Some specializations for performance

def _hstack2d(arr_collection):
    """ faster than `np.hstack` because it avoids calling `np.atleast1d` """
    return np.concatenate(arr_collection, axis=1)

def _hstack1d(arr_collection):
    """ faster than `np.hstack` because it avoids calling `np.atleast1d` """
    return np.concatenate(arr_collection, axis=0)

class coo_matrix_x(sp.coo_matrix):
    """ A COO sparse matrix that assumes that there are no duplicate entries """
    def sum_duplicates(self):
        pass


#--------------------------------------
class SparseCholeskySolver(object):
#--------------------------------------
    """
    Solve a graph for optimal vertex values that satisfy the edge
    constraints using an linearizing least squares solver. Each
    intermediate least squares solution is computed efficiently using a
    sparse cholesky decomposition.

    The state of this solver will be invalidated if the structure of the
    underlying graph changes. In such cases, where the structure of the
    underlying graph changes, create a new solver instance.
    """
    def __init__(self, graph):
        if not hasattr(graph, '_anchor'):
            msg = "SparseCholeskySolver: graph is not anchored." + \
                  "Consider calling Graph.anchor_first_vertex() before solving.\n"
            sys.stderr.write(msg)

        self._graph = graph
        self._sym_decomp_W = None
        self._sym_decomp_JtWJ = None


    def _get_state_update(self):
        edges = self._graph.edges

        residuals = [ e.residual() for e in edges ]

        # For each edge jacobian, compute the its row index in the
        # graph jacobian `J`
        residual_lengths = [ len(r) for r in residuals ]
        row_offsets = [0,] + np.cumsum(residual_lengths).tolist()

        i,j,v = _hstack2d([ e.jacobian(roff=r) for r, e in zip(row_offsets, edges) ])
        J = coo_matrix_x((v, (i,j)))

        Jt = J.T.tocsc()
        J = J.tocsc()

        # Decompose W such that W = U * U.T
        i,j,v = _hstack2d([ e.uncertainty(r,r) for r, e in zip(row_offsets, edges) ])
        W = coo_matrix_x((v, (i, j))).tocsc()
        if self._sym_decomp_W is None:
            self._sym_decomp_W = analyze(W, mode='auto')

        chol_decomp_W = self._sym_decomp_W.cholesky(W)
        U = chol_decomp_W.L()
        JtU = Jt.dot(U)

        # A = J.T * W * J
        #   = J.T * U * U.T * J
        if self._sym_decomp_JtWJ is None:
            self._sym_decomp_JtWJ = analyze_AAt(JtU, mode='auto')

        chol_decomp_JtWJ = self._sym_decomp_JtWJ.cholesky_AAt(JtU)

        r = _hstack1d(residuals)
        b = Jt.dot(W.dot(r))
        x = chol_decomp_JtWJ.solve_A(b)
        return x


    def solve(self, verbose=False, tol=1e-6, maxiter=1000, callback=None):
        current_stats = self._graph.get_stats()

        for iter_ in xrange(maxiter):
            print '  chi2: %.6f    chi2 normalized: %.6f' % current_stats[:2]

            delta = self._get_state_update()
            self._graph.state -= delta

            new_stats = self._graph.get_stats()
            if abs(new_stats.chi2_N - current_stats.chi2_N) < tol:
                break

            current_stats = new_stats


def main():
    np.set_printoptions(precision=4, suppress=True)

    g = load_graph(sys.argv[1] if len(sys.argv) > 1 else 'datasets/MITb.g2o')
    print 'graph has %d vertices, %d edges' % ( len(g.vertices), len(g.edges) )

    g.anchor_first_vertex()

    solver = SparseCholeskySolver(g)
    solver.solve(maxiter=180)

    render_graph_html(g, '/tmp/graph.html')

if __name__ == '__main__':
    main()
