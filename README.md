# stokes-implicit

Under-development implicit time-stepping for the coupled surface kinematical equation and Stokes sub-model problem.

## Firedrake/Python programs

The programs in [`py/`](py/) use [Firedrake](https://www.firedrakeproject.org/), and thus [PETSc](http://www.mcs.anl.gov/petsc/) under the hood.  A brief introduction is in [`py/README.md`](py/README.md).

## old version

I started a different strategy, with fake ice and a Laplace equation domain stretching, in 2020.  There are some ideas there about surface perturbations, and about multilevel solver performance, which I should not lose.  See [`old/`](old/).
