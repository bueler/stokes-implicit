# stokes-implicit

This 3D glacier model combines Stokes momentum balance and coupled, implicitly-computed ice geometry evolution using the surface kinematical equation.  Arbitrary ice geometry and topology changes are allowed as the problem is regarded as a fluid layer evolution subject to a nonnegative thickness inequality constraint (Bueler, 2020).  Conservation of energy, sliding, floating ice, and bedrock motion are _not_ modeled.  The numerical method uses tetrahedral finite elements on an extruded mesh, with stable and aspect-ratio robust Q2 x Q1 mixed elements for the velocity x pressure spaces.  A vertical displacement field is solved-for simultaneously with the Stokes equations; this also uses a Q1 element.

The Python code uses [Firedrake](https://www.firedrakeproject.org/), and thus [PETSc](http://www.mcs.anl.gov/petsc/) under the hood (Bueler, 2021).  Mesh generation is by [Gmsh](http://gmsh.info/) and solution visualization by [Paraview](https://www.paraview.org/).

## installation

  * Install [Gmsh](http://gmsh.info/) and [Paraview](https://www.paraview.org/), perhaps by installing packages.
  * Follow the instructions at the [Firedrake download page](https://www.firedrakeproject.org/download.html) to install Firedrake.
  * Most users will only need the PETSc which is installed by Firedrake.  A separate PETSc installation may be less convenient.

## usage

The solver default mode uses a dome geometry computed from the Halfar (1981,1983) solutions of the shallow ice approximation.  Thus no mesh-generation is needed.

For the simplest version start Firedrake and run the solver to compute a single time step:

        $ source ~/firedrake/bin/activate
        (firedrake) $ ./stokesi.py -o dome.pvd

This 2D computation uses an extruded mesh with only 30x4=120 quadrilateral elements.  It writes variables (velocity,pressure,displacement) into `dome.pvd`.  Visualize:

        (firedrake) $ paraview dome.pvd

The following parallel 3D computation uses an extruded mesh with 30x30x8=7200 tetrahedral elements and N=207k degrees of freedom.  It takes a few minutes:

        (firedrake) $ mpiexec -n 6 ./stokesi.py -mx 30 -my 30 -vertrefine 1 -s_snes_rtol 1.0e-3 -s_snes_monitor -s_ksp_converged_reason -saveextra -o dome3.pvd

Note that rich diagnostic information is generated with `-saveextra` including the full deviatoric stress tensor and also (for comparison) the SIA velocities.

FIXME Generate some other domain geometry and the mesh:

        $ gmsh -3 glacier.geo

## performance

FIXME multigrid schemes based on semicoarsening; see fixed-boundary analog in `pool.py`

## references

  * There is a related Firedrake Stokes model with explicitly-updated geometry in my McCarthy Summer School notes repository [mccarthy/stokes/](https://github.com/bueler/mccarthy/tree/master/stokes).
  * [E. Bueler, _PETSc for Partial Differential Equations_, SIAM Press 2021](https://my.siam.org/Store/Product/viewproduct/?ProductId=32850137)
  * E. Bueler, _Conservation laws for free-boundary fluid layers_, 2020, [arxiv:2007.05625](https://arxiv.org/abs/2007.05625)

