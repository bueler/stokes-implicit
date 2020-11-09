# stokes-implicit/py/

The Python codes in this directory use [Firedrake](https://www.firedrakeproject.org/), and thus [PETSc](http://www.mcs.anl.gov/petsc/) under the hood.  Mesh generation is by [Gmsh](http://gmsh.info/) and solution visualization by [Paraview](https://www.paraview.org/).

## installation

  * Install [Gmsh](http://gmsh.info/) and [Paraview](https://www.paraview.org/), perhaps by installing packages.
  * Follow the instructions at the [Firedrake download page](https://www.firedrakeproject.org/download.html) to install Firedrake.

## usage

The solver default mode uses a dome geometry computed from the Halfar (1981,1983) solutions of the shallow ice approximation.  Thus no mesh-generation is needed.

For the simplest version start Firedrake and run the solver to compute a single time step:

        $ source ~/firedrake/bin/activate
        (firedrake) $ ./stokesi.py -o dome.pvd

This 2D computation uses an extruded mesh with 30x4=120 quadrilateral elements.  It writes variables (velocity,pressure,displacement) into `dome.pvd`.  Visualize the result this way:

        (firedrake) $ paraview dome.pvd

The following parallel 3D computation uses an extruded mesh with 30x30x8=7200 tetrahedral elements and N=207k degrees of freedom.  It takes a few minutes:

        (firedrake) $ mpiexec -n 8 ./stokesi.py -mx 30 -my 30 -refine 1 -s_snes_rtol 1.0e-3 -s_snes_monitor -s_ksp_converged_reason -saveextra -o dome3.pvd

Note that rich diagnostic information is generated with `-saveextra` including the full deviatoric stress tensor and also (for comparison) the SIA velocities.

## performance

FIXME multigrid schemes based on semicoarsening; see fixed-boundary analog in `pool.py`

For a reminder of how fast it is for Poisson in 2D:
        (firedrake) ~/repos/p4pdes/python/ch13[master]$ tmpg -n 8 ./fish.py -refine 12 -quad -s_ksp_converged_reason -s_pc_type mg -s_mg_levels_ksp_type richardson -s_mg_levels_pc_type bjacobi -s_mg_levels_sub_pc_type icc -s_ksp_rtol 1.0e-10
          Linear s_ solve converged due to CONVERGED_RTOL iterations 6
        done on 8193 x 8193 grid with Q_1 elements:
          error |u-uexact|_inf = 6.034e-10, |u-uexact|_h = 2.631e-10
        real 651.31

FIXME paper title:  How fast are 3D Stokes solvers for ice sheets?: Performance analysis and expectations.


