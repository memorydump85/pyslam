#! /usr/bin/python

import numpy as np
import sys

print '# num attributes'
print '0'

f = open(sys.argv[1])
for line in f:
    d = line.split()

    i = lambda idx: int(d[idx])
    f = lambda idx: float(d[idx])

    if d[0] == 'VERTEX_SE2':
        id_ = i(1)
        print '# node %d' % id_
        print '"april.graph.GXYTNode"'
        print '{'
        print '  # state'
        print '  vec 3'
        print '  %f %f %f' % (f(2), f(3), f(4))
        print '  # truth'
        print '  vec -1'
        print '  # initial value'
        print '  vec 3'
        print '  %f %f %f' % (f(2), f(3), f(4))
        print '  # num attributes'
        print '  0'
        print '}'

    elif d[0] == 'EDGE_SE2':
        ndx_out, ndx_in = i(1), i(2)
        P = np.array([ [ f(6), f( 7), f( 8) ],
                       [ f(7), f( 9), f(10) ],
                       [ f(8), f(10), f(11) ] ])
        C = np.linalg.inv(P).ravel()
        print '"april.graph.GXYTEdge"'
        print '{'
        print '  # a, b'
        print '  %d' % i(1)
        print '  %d' % i(2)
        print '  # XYT'
        print '  vec 3'
        print '  %f %f %f' % (f(3), f(4), f(5))
        print '  # XYT truth'
        print '  vec -1'
        print '  # Covariance'
        print '  mat 3 3'
        print '  %f %f %f' % (C[0], C[1], C[2])
        print '  %f %f %f' % (C[3], C[4], C[5])
        print '  %f %f %f' % (C[6], C[7], C[8])
        print '  # num attributes'
        print '  0'
        print '}'

    print ''
