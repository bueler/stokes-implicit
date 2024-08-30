# stokes-implicit

Under-development implicit time-stepping for the coupled surface kinematical equation and Stokes sub-model problem.

## Firedrake/Python programs

The programs which will be in [`py/`](py/) use [Firedrake](https://www.firedrakeproject.org/), and thus [PETSc](http://www.mcs.anl.gov/petsc/) under the hood (Bueler, 2021).  A brief introduction is in [`py/README.md`](py/README.md), when I write that.

## old version

I started a different strategy, with fake ice and a Laplace equation domain stretching, in 2020.  See `old/`.
