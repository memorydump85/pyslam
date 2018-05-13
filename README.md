pyslam
======
Basic Sparse-Cholesky Graph SLAM solver implemented in python

Requirements
------------
* numpy >= 1.12
* scipy >= 0.18
* cython >= 0.25
* scikit-sparse >= 0.4.2
* suitesparse-dev

On Ubuntu 16.04:
```bash
apt-get install libsuitesparse-dev
pip install future numpy scipy cython scikit-sparse
```

Invocation
----------
```bash
python slam_solver.py datasets/M3500a.g2o
```

The resulting optimized SLAM graph is rendered as HTML and saved in
`/tmp/graph.html`