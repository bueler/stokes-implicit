# stokes-implicit

This 3D glacier model combines Stokes momentum balance and coupled, implicitly-computed ice geometry evolution using the surface kinematical equation.  Arbitrary ice geometry and topology changes are allowed as the problem is regarded as a fluid layer evolution subject to a nonnegative thickness inequality constraint (Bueler, 2020).  Conservation of energy, sliding, floating ice, and bedrock motion are _not_ modeled.  The numerical method uses tetrahedral finite elements on an extruded mesh, with stable and aspect-ratio robust Q2 x Q1 mixed elements for the velocity x pressure spaces.  A vertical displacement field is solved-for simultaneously with the Stokes equations; this also uses a Q1 element.

The current project includes a draft paper in [`doc/`](doc/) and Python codes in [`py/`](py/).  These codes use [Firedrake](https://www.firedrakeproject.org/), and thus [PETSc](http://www.mcs.anl.gov/petsc/) under the hood (Bueler, 2021).  For a brief introduction to using these codes see [`py/README.md`](py/README.md)

## references

  * [E. Bueler, _PETSc for Partial Differential Equations_, SIAM Press 2021](https://my.siam.org/Store/Product/viewproduct/?ProductId=32850137)
  * E. Bueler, _Conservation laws for free-boundary fluid layers_, 2020, [arxiv:2007.05625](https://arxiv.org/abs/2007.05625)
  * An earlier Firedrake Stokes model with explicitly-updated geometry appears in my McCarthy Summer School notes repository [mccarthy/stokes/](https://github.com/bueler/mccarthy/tree/master/stokes).

