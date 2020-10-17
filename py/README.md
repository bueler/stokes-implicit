# stokes-implicit/py/

The Python codes in this directory use [Firedrake](https://www.firedrakeproject.org/), and thus [PETSc](http://www.mcs.anl.gov/petsc/) under the hood.  Mesh generation is by [Gmsh](http://gmsh.info/) and solution visualization by [Paraview](https://www.paraview.org/).

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

