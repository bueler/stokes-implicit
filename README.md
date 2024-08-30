# stokes-implicit

Under-development implicit time-stepping for the coupled surface kinematical equation and Stokes sub-model problem.

## Firedrake/Python programs

The programs in [`py/`](py/) use [Firedrake](https://www.firedrakeproject.org/), and thus [PETSc](http://www.mcs.anl.gov/petsc/) under the hood.  A brief introduction is in [`py/README.md`](py/README.md).

These programs currently have a weak dependence on the library in [github.com/bueler/stokes-extrude](https://github.com/bueler/stokes-extrude), but that library will need major revisions to include the ideas under development here.

## documentation

Some ideas here came from [github.com/bueler/glacier-fe-estimate](https://github.com/bueler/glacier-fe-estimate), which is not a finished project, though it has reached the preprint stage.  For now that [arxiv preprint](https://arxiv.org/abs/2408.06470) is the closest to documentation of the intention here.

## old versions

I started a different strategy, with fake ice and a Laplace equation domain stretching, in 2020.  That extended to 2021 only, and now it is abandoned.  The old material includes some ideas about surface perturbations and multilevel solver performance which I should not lose.  See [`old/`](old/) in the current repo.

See also the probably-abandoned work in [github.com/bueler/multilevel-stokes-geometry](https://github.com/bueler/multilevel-stokes-geometry), which is a mostly-2021 project using semi-coarsened mesh hierarchy added to Firedrake by Lawrence Mitchell.

Finally I maintain a [tutorial on using Firedrake to solve the glaciological Stokes problem on a fixed domain](https://github.com/bueler/stokes-ice-tutorial), which is in yet another repo.