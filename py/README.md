# stokes-implicit/py/

## requirements

1. Python3 is required.

2. [Download and install Firedrake.](https://www.firedrakeproject.org/download.html)

3. Clone and install `stokes-extrude` using [pip](https://pypi.org/project/pip/):

```bash
$ git clone https://github.com/bueler/stokes-extrude.git
$ cd stokes-extrude/
$ pip install -e .
```

## basic usage on single step

3D glacier (2D horizontal) with 1 year time step:

    $ python3 case.py 20 10 1 2

2D glacier (1D horizontal) with 1 year time step:

    $ python3 case.py 80 20 1 1

These run in parallel too, e.g.

    $ mpiexec -n 4 python3 case.py 20 10 1 2

FIXME

## FSSA, symmetric FSSA, and edge stabilization

For classical FSSA:

    $ python3 case.py 200 20 1 1    # 1 year run is visually stable

    $ python3 case.py 200 20 3 1    # 3 year run has obvious wiggles

## regarding performance

See `solverparams.py` in stokes-extrude

See https://www.firedrakeproject.org/demos/stokes.py.html

See https://www.firedrakeproject.org/petsc-interface.html
