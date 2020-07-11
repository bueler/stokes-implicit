# stokes-implicit

This is a planned glacier model combining Stokes momentum balance and coupled,
implicitly-updated geometry.  It is started from the Firedrake Stokes model with
explicitly-updated geometry described in my [mccarthy/stokes/](https://github.com/bueler/mccarthy/tree/master/stokes) repo.

The code uses [Firedrake](https://www.firedrakeproject.org/), and thus [PETSc](http://www.mcs.anl.gov/petsc/) under the hood.  Mesh generation is by [Gmsh](http://gmsh.info/) and solution visualization by [Paraview](https://www.paraview.org/).

## installation

  * Install [Gmsh](http://gmsh.info/) and [Paraview](https://www.paraview.org/),
    for instance by installing Debian or OSX packages.
  * Follow the instructions at the
    [Firedrake download page](https://www.firedrakeproject.org/download.html)
    to install Firedrake.
  * Most users will only need the PETSc which is installed by Firedrake.  Any
    separate PETSc installation should (generally) not be used in the Firedrake
    install.

## usage

FIXME Generate the domain geometry and the mesh:

        $ ./gendomain.py -o glacier.geo  # domain outline
        $ gmsh -2 glacier.geo            # mesh domain; generates glacier.msh

FIXME Start Firedrake and run the solver:

        $ source ~/firedrake/bin/activate
        (firedrake) $ ./flow.py glacier.msh

FIXME This writes variables (velocity,pressure) into `glacier.pvd`.  Now visualize:

        (firedrake) $ paraview glacier.pvd

