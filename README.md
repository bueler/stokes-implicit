# stokes-implicit

This 3D glacier model combines Stokes momentum balance and coupled, implicitly-computed ice geometry evolution using the surface kinematical equation.  Arbitrary ice geometry and topology changes are allowed as the problem is regarded as a fluid layer evolution subject to a nonnegative thickness inequality constraint (Bueler, 2020).  Conservation of energy, sliding, floating ice, and bedrock motion are _not_ modeled.  The numerical method uses Q2 x Q1 (velocity x pressure) finite elements on a 2D or 3D extruded mesh.  A vertical displacement field is solved-for simultaneously with the Stokes equations; this also uses a Q1 element.

The current project includes a draft paper `simp.tex` in [`paper/`](paper/) and Python programs in [`py/`](py/).

## Firedrake/Python programs

These programs use [Firedrake](https://www.firedrakeproject.org/), and thus [PETSc](http://www.mcs.anl.gov/petsc/) under the hood (Bueler, 2021).  A brief introduction is in [`py/README.md`](py/README.md)

## references

See the references cited in `simp.tex`.  In addition, an earlier Firedrake Stokes model with explicitly-updated geometry appears in my McCarthy Summer School notes repository [mccarthy/stokes/](https://github.com/bueler/mccarthy/tree/master/stokes).

  * E. Bueler, _Conservation laws for free-boundary fluid layers_, 2020, [arxiv:2007.05625](https://arxiv.org/abs/2007.05625)

