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

## running the Stokes + SKE solver

    $ python3 run.py BDIM MX MZ NSTEPS DTYEAR

## basic usage

3D glacier (2D horizontal) with single 1 year time step:

    $ python3 run.py 2 20 10 1 1.0

2D glacier (1D horizontal) with 1 year time step:

    $ python3 run.py 1 80 20 1 1.0

These run in parallel, e.g.

    $ mpiexec -n 4 python3 run.py 2 20 10 1 1.0

FIXME

## FSSA, symmetric FSSA, and edge stabilization

For classical (2022) FSSA:

    $ python3 run.py 1 200 20 1 1.0    # 1 year run is visually stable

    $ python3 run.py 1 200 20 1 3.0    # 3 year run has obvious wiggles

## regarding performance

See `solverparams.py` in stokes-extrude

See https://www.firedrakeproject.org/demos/stokes.py.html

See https://www.firedrakeproject.org/petsc-interface.html
