#!/usr/bin/env python3

# TODO:
#   * change regularization so that instead of j() having a discontinuous derivative, the top boundary value of c never reaches -Href
#   * save the new domain into a separate file so I can look at it
#   * combine solver parameters into something like the packages in mccarthy/stokes/
#   * initialize with u=(SIA velocity), p=(hydrostatic) and c=0
#   * option -sialaps N: do SIA evals N times and quit; for timing; defines work unit
#   * get semicoarsening to work with mg; use pool.py as testing ground
#   * test nonlinear full MG cycle for semicoarsening (i.e. roll -snes_grid_sequence by hand) using (p - rho g depth)^2 penalty on coarser

# serial 2D example: runs in about two minutes with 5/1 element ratio and N=8.2e4
# timer ./stokesi.py -dta 0.01 -s_snes_converged_reason -s_ksp_converged_reason -s_snes_rtol 1.0e-4 -mx 960 -refine 1 -saveextra -o foo2.pvd

# parallel 3D example: runs in about 10 minutes with 80/1 element ratio and N=1.1e5
# tmpg -n 8 ./stokesi.py -dta 0.01 -s_snes_converged_reason -s_ksp_converged_reason -s_snes_rtol 1.0e-4 -mx 30 -my 30 -saveextra -o foo3.pvd

import sys,argparse
from firedrake import *
from src.constants import secpera, rho
from src.meshes import basemesh, extrudedmesh, deformlimitmesh
from src.spaces import vectorspaces
from src.halfar import t0_2d, t0_3d, halfar_2d, halfar_3d
from src.functionals import IceModel, IceModel2D
from src.diagnostic import writeresult

parser = argparse.ArgumentParser(description='''
Solve coupled Glen-Stokes equations plus surface kinematical equation (SKE)
for a glacier or ice sheet.  Generates flat-bed 2D or 3D mesh by extrusion
of an equally-spaced interval or quadrilateral mesh in the map plane
giving quadrilateral or hexahedral elements, respectively.  Optional
refinement in the vertical to allow semicoarsening multigrid.  Currently the
initial geometry is from Halfar (1981) or Halfar (1983).  A reference
domain with a minimum thickness is generated from the initial geometry.
We solve a nonlinear system for velocity u, pressure p, and (scalar)
vertical displacement c.  The system of 3 PDEs corresponds to a single
backward Euler time step of -dta years:
  stress balance:       F_1(u,p,c) = 0
  incompressibility:    F_2(u,c)   = 0
  Laplace/SKE:          F_3(u,c)   = 0
The last equation is Laplace's equation for c in the domain interior but it
is coupled to the first two through the top boundary condition which enforces
the SKE and through weighting/stretching factors in the Glen-Stokes weak form.
The mixed space consists of (u,p) in Q2 x Q1 for the Stokes problem and c in
Q1 for displacement and thus the Jacobian matrices have block form
      * * | *
  J = *   | *
      -------
      *   | *
where the upper-left 2x2 (u,p) block is a weighted form of the usual mixed and
element Stokes matrix.  The default solver is multiplicative fieldsplit between
the (u,p) block and the c blocks.  The (u,p) block is solved by Schur lower
fieldsplit with selfp preconditioning on its Schur block.  By default the
diagonal blocks are solved (preconditioned) by LU using MUMPS.''',
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 add_help=False)
parser.add_argument('-almost', action='store_true', default=False,
                    help='use the draft weak forms which do not fully couple')
parser.add_argument('-dirichletsmb', action='store_true', default=False,
                    help='apply simplified SMB condition on top of reference domain')
parser.add_argument('-dta', type=float, default=0.01, metavar='X',
                    help='length of time step in years (default=0.01 a)')
parser.add_argument('-Dtyp', type=float, default=2.0, metavar='X',
                    help='typical strain rate in "+(eps Dtyp)^2" (default=2.0 a-1)')
parser.add_argument('-eps', type=float, default=0.0001, metavar='X',
                    help='to regularize viscosity by "+eps Dtyp^2" (default=0.0001)')
parser.add_argument('-H0', type=float, default=3000.0, metavar='X',
                    help='center height in m of initial ice sheet (default=3000)')
parser.add_argument('-Href', type=float, default=200.0, metavar='X',
                    help='minimum thickness in m of reference domain (default=200)')
parser.add_argument('-L', type=float, default=60.0e3, metavar='X',
                    help='half-width in m of computational domain (default=60e3)')
parser.add_argument('-mx', type=int, default=30, metavar='N',
                    help='number of equal subintervals in x-direction (default=30)')
parser.add_argument('-my', type=int, default=-1, metavar='N',
                    help='3D solve if my>0: subintervals in y-direction (default=-1)')
parser.add_argument('-mz', type=int, default=4, metavar='N',
                    help='number of layers in each vertical column (default=4)')
parser.add_argument('-o', metavar='FILE.pvd', type=str, default='',
                    help='save to output file name ending with .pvd')
parser.add_argument('-pvert', type=int, default=0, metavar='N',
                    help='p-refinement level in vertical; use 0,1,2,3 only (default=0)')
parser.add_argument('-R0', type=float, default=50.0e3, metavar='X',
                    help='half-width in m of initial ice sheet (default=50e3)')
parser.add_argument('-refine', type=int, default=-1, metavar='N',
                    help='number of vertical (z) mesh refinement levels')
parser.add_argument('-saveextra', action='store_true', default=False,
                    help='add stresses and SIA velocity to -o output file')
parser.add_argument('-stokesihelp', action='store_true', default=False,
                    help='print help for stokesi.py and quit')
args, unknown = parser.parse_known_args()
if args.stokesihelp:
    parser.print_help()
    sys.exit(0)

# are we 3D or 2D
ThreeD = args.my > 0

# correct units to SI
Dtyp = args.Dtyp / secpera
dt = args.dta * secpera

# set up base mesh; note refinement is NOT used here
base_mesh = basemesh(L=args.L,mx=args.mx,my=args.my)

# report on base mesh
PETSc.Sys.Print('**** SUMMARY OF SETUP ****')
if ThreeD:
    PETSc.Sys.Print('horizontal domain:   [%.2f,%.2f] x [%.2f,%.2f] km square'
                    % (-args.L/1000.0,args.L/1000.0,-args.L/1000.0,args.L/1000.0))
    PETSc.Sys.Print('base mesh:           %d x %d elements (quads)' % (args.mx,args.my))
else:
    PETSc.Sys.Print('horizontal domain:   [%.2f,%.2f] km interval'
                    % (-args.L/1000.0,args.L/1000.0))
    PETSc.Sys.Print('base mesh:           %d elements (intervals)' % args.mx)

def deformbyinitial(mesh):
    '''Use initial shape to determine mesh for reference domain.
    Currently this just uses the Halfar solution.'''
    if ThreeD:
        x,y,_ = SpatialCoordinate(mesh)
        Hinitial = halfar_3d(x,y,R0=args.R0,H0=args.H0)
    else:
        x,_ = SpatialCoordinate(mesh)
        Hinitial = halfar_2d(x,R0=args.R0,H0=args.H0)
    deformlimitmesh(mesh,Constant(0.0),Hinitial,Href=args.Href)
    return Hinitial

# FIXME need time-stepping loop, which will alter mesh at every step

# extrude mesh, generating hierarchy if refining, and deform to match initial
# shape, but limited at Href
if args.refine > 0:
    mesh, hierarchy = extrudedmesh(base_mesh, args.mz, refine=args.refine)
    for kmesh in hierarchy:
        hinitialextruded = deformbyinitial(kmesh)
    mzfine = args.mz * 2**args.refine
    PETSc.Sys.Print('refined vertical:    %d coarse layers refined to %d fine layers' \
                    % (args.mz,mzfine))
else:
    mesh = extrudedmesh(base_mesh, args.mz)
    hinitialextruded = deformbyinitial(mesh)
    mzfine = args.mz

# extruded mesh coordinates
if ThreeD:
    x,y,z = SpatialCoordinate(mesh)
else:
    x,z = SpatialCoordinate(mesh)

# report on generated geometry and extruded mesh
dxelem = 2.0 * args.L / args.mx
dzrefelem = args.Href / mzfine
if ThreeD:
    dyelem = 2.0 * args.L / args.my
    PETSc.Sys.Print('initial condition:   3D Halfar, H0=%.2f m, R0=%.3f km, t0=%.5f a'
                    % (args.H0,args.R0/1000.0,t0_3d(args.R0,args.H0)/secpera))
    PETSc.Sys.Print('3D extruded mesh:    %d x %d x %d elements (hexahedra); limited at Href=%.2f m'
                    % (args.mx,args.my,mzfine,args.Href))
    PETSc.Sys.Print('element dimensions:  dx=%.2f m, dy=%.2f m, dz_min=%.2f m, ratiox=%.1f, ratioy=%.1f'
                    % (dxelem,dyelem,dzrefelem,dxelem/dzrefelem,dyelem/dzrefelem))
else:
    PETSc.Sys.Print('initial condition:   2D Halfar, H0=%.2f m, R0=%.3f km, t0=%.5f a'
                    % (args.H0,args.R0/1000.0,t0_2d(args.R0,args.H0)/secpera))
    PETSc.Sys.Print('2D extruded mesh:    %d x %d elements (quads); limited at Href=%.2f m'
                    % (args.mx,mzfine,args.Href))
    PETSc.Sys.Print('element dimensions:  dx=%.2f m, dz_min=%.2f m, ratio=%.1f'
                    % (dxelem,dzrefelem,dxelem/dzrefelem))

# set up mixed finite element space
Vu, Vp, Vc = vectorspaces(mesh,vertical_higher_order=args.pvert)
Z = Vu * Vp * Vc

# report on vector spaces sizes
n_u,n_p,n_c,N = Vu.dim(),Vp.dim(),Vc.dim(),Z.dim()
PETSc.Sys.Print('vector space dims:   n_u=%d, n_p=%d, n_c=%d  -->  N=%d' \
                % (n_u,n_p,n_c,N))

# trial and test functions
upc = Function(Z)
u,p,c = split(upc)
v,q,e = TestFunctions(Z)

# coupled weak form
if ThreeD:
    im = IceModel(almost=args.almost, mesh=mesh,
                  Href=args.Href, eps=args.eps, Dtyp=Dtyp,
                  hcurrent=hinitialextruded)
else:
    im = IceModel2D(almost=args.almost, mesh=mesh,
                    Href=args.Href, eps=args.eps, Dtyp=Dtyp,
                    hcurrent=hinitialextruded)
F = im.F(u,p,c,v,q,e)

# apply surface kinematical equation by adding to weak form
a = Constant(0.0) # FIXME only correct for Halfar
if not args.dirichletsmb:
    F += im.Fsmb(a,dt,u,c,e)

# boundary conditions
zerovelocity = Constant((0.0, 0.0, 0.0)) if ThreeD else Constant((0.0, 0.0))
sides = (1,2,3,4) if ThreeD else (1,2)
bcs = [DirichletBC(Z.sub(0), zerovelocity, 'bottom'),
       DirichletBC(Z.sub(0), zerovelocity, sides),
       DirichletBC(Z.sub(2), Constant(0.0), 'bottom')]  # for displacement
if args.dirichletsmb: # artificial, for testing
    # set Dirichlet condition on top
    bcs.append(DirichletBC(Z.sub(2), im.smbref(a,dt,z), 'top'))

# solver parameters; some are defaults which are deliberately made explicit here
# note 'lu' = mumps, both in serial and parallel (faster)
parameters = {'snes_linesearch_type': 'bt',  # new firedrake default is "basic", i.e. NO linesearch
              'mat_type': 'aij',
              'ksp_type': 'gmres',  # consider fgmres and -fieldsplit_0_ksp_type gmres fieldsplit_0_ksp_max_it 3
              'ksp_pc_side': 'right',  # consider left
              'pc_type': 'fieldsplit',
              'pc_fieldsplit_type': 'symmetric_multiplicative',  # 'multiplicative' or 'additive': more linear iters
              'pc_fieldsplit_0_fields': '0,1',
              'pc_fieldsplit_1_fields': '2',
              # schur fieldsplit for (u,p)-(u,p) block
              'fieldsplit_0_ksp_type': 'preonly',
              'fieldsplit_0_pc_type': 'fieldsplit',
              'fieldsplit_0_pc_fieldsplit_type': 'schur',
              'fieldsplit_0_pc_fieldsplit_schur_fact_type': 'lower',
              # selfp seems to be faster than a Mass object
              'fieldsplit_0_pc_fieldsplit_schur_precondition': 'selfp',
              # LU on the u-u block  (neither AMG nor mg work anywhere near as fast for now)
              'fieldsplit_0_fieldsplit_0_ksp_type': 'preonly',
              'fieldsplit_0_fieldsplit_0_pc_type': 'lu',
              #'fieldsplit_0_fieldsplit_0_pc_type': 'gamg',
              #'fieldsplit_0_fieldsplit_0_pc_gamg_type': 'agg',
              #'fieldsplit_0_fieldsplit_0_mg_levels_ksp_type': 'chebyshev',
              #'fieldsplit_0_fieldsplit_0_mg_levels_pc_type': 'sor',
              'fieldsplit_0_fieldsplit_1_pc_type': 'jacobi',
              'fieldsplit_0_fieldsplit_1_pc_jacobi_type': 'diagonal',
              # LU on the c-c block; AMG not great; mg fails with zero row msg; hypre (w/o tuning) seems slower
              # classical few iters and faster than agg (but grid complexity better for agg)
              'fieldsplit_1_ksp_type': 'preonly',
              'fieldsplit_1_pc_type': 'lu'}
              #'fieldsplit_1_pc_type': 'gamg',
              #'fieldsplit_1_pc_gamg_type': 'classical',
              #'fieldsplit_1_pc_gamg_square_graph': '1'}

## PLAYING AROUND:
#              'fieldsplit_1_pc_type': 'mg',
#              'fieldsplit_1_pc_mg_levels': args.vertrefine+1,
#              'fieldsplit_1_pc_mg_galerkin': None,
#              'fieldsplit_1_mg_levels_ksp_type': 'richardson',
#              'fieldsplit_1_mg_levels_ksp_max_it': 2,
#              'fieldsplit_1_mg_levels_pc_type': 'lu',
#              'fieldsplit_1_mg_coarse_ksp_type': 'preonly',
#              'fieldsplit_1_mg_coarse_pc_type': 'lu'}

# mumps seems to be slower than PETSc lu in serial (I'm confused) and this
# conditional may just be the default
if base_mesh.comm.size > 1:
    parameters['fieldsplit_0_fieldsplit_0_pc_factor_mat_solver_type'] = 'mumps'
    parameters['fieldsplit_1_pc_factor_mat_solver_type'] = 'mumps'

# solve system as though it is nonlinear:  F(u) = 0
PETSc.Sys.Print('**** SOLVING ****')
PETSc.Sys.Print('    ... one time step dt=%.5f a ...' % args.dta)
solve(F == 0, upc, bcs=bcs, options_prefix = 's',
      solver_parameters=parameters)

# save ParaView-readable file
if args.o:
    writeresult(args.o,mesh,im,upc,hinitialextruded,saveextra=args.saveextra)

