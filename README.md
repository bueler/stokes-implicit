# stokes-implicit

Under-development implicit time-stepping for the coupled surface kinematical equation and Stokes sub-model problem.

## Firedrake/Python programs

The programs in [`py/`](py/) use [Firedrake](https://www.firedrakeproject.org/), and thus [PETSc](http://www.mcs.anl.gov/petsc/) under the hood.  A brief introduction is in [`py/README.md`](py/README.md).

## old version

I started a different strategy, with fake ice and a Laplace equation domain stretching, in 2020.  That extended to 2021 only, and now it is abandoned.  The old material includes some ideas about surface perturbations and multilevel solver performance which I should not lose.  See [`old/`](old/).
