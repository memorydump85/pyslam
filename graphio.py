import numpy as np

from graph import Graph, VertexXYT, EdgeXYT
from mmath import MultiVariateGaussian



class GraphIOError(Exception):
    pass


def load_graph_g2o(filename):
    from collections import OrderedDict
    vertices = OrderedDict()
    edges = []

    with open(filename) as f:
        for linenum, line in enumerate(f):
            d = line.split()

            i = lambda idx: int(d[idx])
            f = lambda idx: float(d[idx])

            if d[0] == 'VERTEX_SE2':
                id_ = i(1)
                xyt = np.array([ f(2), f(3), f(4) ])
                vertices[id_] = VertexXYT(xyt)

            elif d[0] == 'EDGE_SE2':
                ndx_out, ndx_in = i(1), i(2)
                v_out, v_in = vertices[ndx_out], vertices[ndx_in]

                xyt = np.array([ f(3), f( 4), f( 5) ])
                C = np.array([ [ f(6), f( 7), f( 8) ],
                               [ f(7), f( 9), f(10) ],
                               [ f(8), f(10), f(11) ] ])
                g = MultiVariateGaussian(xyt, np.linalg.inv(C))
                edges.append( EdgeXYT(v_out, v_in, g) )

            else:
                msg = "Unknown edge or vertex type %s in line %d" % (d[0], linenum)
                raise GraphIOError(msg)

    return Graph(vertices.values(), edges)


def load_graph_toro(filename):
    from collections import OrderedDict
    vertices = OrderedDict()
    edges = []

    with open(filename) as f:
        for linenum, line in enumerate(f):
            d = line.split()

            i = lambda idx: int(d[idx])
            f = lambda idx: float(d[idx])

            if d[0] == 'VERTEX2':
                id_ = i(1)
                xyt = np.array([ f(2), f(3), f(4) ])
                vertices[id_] = VertexXYT(xyt)

            elif d[0] == 'EDGE2':
                ndx_out, ndx_in = i(1), i(2)
                v_out, v_in = vertices[ndx_out], vertices[ndx_in]

                xyt = np.array([ f( 3), f( 4), f( 5) ])
                C = np.array([ [ f( 6), f( 7), f(10) ],
                               [ f( 7), f( 8), f(11) ],
                               [ f(10), f(11), f( 9) ] ])
                g = MultiVariateGaussian(xyt, C)
                edges.append( EdgeXYT(v_out, v_in, g) )

            else:
                msg = "Unknown edge or vertex type %s in line %d" % (d[0], linenum)
                raise Exception(msg)

    return Graph(vertices.values(), edges)


def load_graph(filename):
    if filename.endswith('.g2o'):
        return load_graph_g2o(filename)
    elif filename.endswith('.toro'):
        return load_graph_toro(filename)
    else:
        raise GraphIOError('Unsupported file extension: ' % filename)


def render_graph_svg(graph, filename=None):
    doc_template = \
    """\
    <html style="overflow: scroll">
    <svg viewbox="-1500 -1500 3000 3000">
      <defs>
        <g id="R" transform="translate(-5 -5) rotate(-45)">
            <rect x="5" y="5" width="5" height="5" fill="steelblue" />
            <circle cx="5" cy="5" r="5" fill="steelblue" />
        </g>
      </defs>\n{content}
    </svg>
    </html>
    """

    vertex_template = \
    """\
      <use href="#R" x="{x:.3f}" y="{y:.3f}" transform="rotate({t:.3f} {x:.3f} {y:.3f})" />"""

    svg_content = []

    vertex_xyts = (v.state for v in graph.vertices if isinstance(v, VertexXYT))
    transforms = ( dict(x=v[0]*5, y=v[1]*5, t=np.degrees(v[2])) for v in vertex_xyts )
    svg_content = [ vertex_template.format(**xyt) for xyt in transforms ]

    svg_content = "\n".join(svg_content)
    doc = doc_template.format(content=svg_content)

    import textwrap
    doc = textwrap.dedent(doc)

    if filename is None:
        return doc
    else:
        with open(filename, 'w') as f:
            f.write(doc)