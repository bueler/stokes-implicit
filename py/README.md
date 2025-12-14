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

FIXME

