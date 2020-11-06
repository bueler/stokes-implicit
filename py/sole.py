#!/usr/bin/env python3

# notes: compare the following 5 solvers, all of which give 16x16x16 meshes of Q1 hexahedra:
#   ./sole.py -refine 4 -stage 1
#   ./sole.py -mx 16 -my 16 -refine 4 -stage 2
#   ./sole.py -mx 16 -my 16 -refine 2 -stage 2 -aggressive
#   ./sole.py -mx 16 -my 16 -refine 4 -stage 3
#   ./sole.py -mx 16 -my 16 -refine 2 -stage 3 -aggressive

# result: on 128x128x128 meshes, higher performance from aggressive semi-coarsening:
#   $ tmpg -n 1 ./sole.py -refine 7 -stage 1               # for reference: 3D GMG refinement
#     Linear s_ solve converged due to CONVERGED_RTOL iterations 2
#     GMG levels = 8
#   real 130.58
#   $ tmpg -n 1 ./sole.py -mx 128 -my 128 -mz 2 -refine 6 -stage 3
#     Linear s_ solve converged due to CONVERGED_RTOL iterations 3
#     GMG levels = 7, coarse-level AMG levels = 4
#   real 226.49
#   $ tmpg -n 1 ./sole.py -mx 128 -my 128 -mz 2 -refine 3 -stage 3 -aggressive
#     Linear s_ solve converged due to CONVERGED_RTOL iterations 3
#     GMG levels = 4, coarse-level AMG levels = 4          # same # of levels as 3D GMG
#   real 151.58                                            # only 16% loss over 3D GMG

# result: high performance on a ice-sheet relevant grid aspect, with small discretization error;
# note runtime is less than 2 minutes for N=8.7e7 degrees of freedom; achieved here without matrix-free
#   $ tmpg -n 8 ./sole.py -mx 512 -my 512 -mz 2 -refine 2 -stage 3 -aggressive -s_ksp_rtol 1.0e-10
#   extruded mesh:      512 x 512 x 32 mesh of Q1 elements on 1.00 x 1.00 x 0.01 domain
#   vector space dim:   N=8684577
#     Linear s_ solve converged due to CONVERGED_RTOL iterations 8
#     GMG levels = 3, coarse-level AMG levels = 5
#     L_2 error norm = 7.07559e-06
#   real 104.97


from firedrake import *
import sys, argparse
from pprint import pprint

parser = argparse.ArgumentParser(description='''
Three stages of multigrid Poisson solvers in 3D, always using regular meshes
with hexahedra.  Stage 3 is a high-aspect ratio box with small height.
(Note that a sole is a kind of flat fish.)  Set the coarsest grid with
-mx, -my, -mz; defaults are (mx,my,mz)=(1,1,1) and the default -refine is 0.
For stage 1 -refine acts equally in all dimensions and the solution is ordinary
GMG.  For stages 2,3 the GMG refinement is only in z (semi-coarsening) and the
coarse grid problem is solved by AMG (-mg_coarse_pc_type gamg).  In stages 2,3
the semi-coarsening uses a default factor of 2 but with -aggressive this is 4.
The plan is to have the first three stages in pool.py be analogs of here,
but for Stokes problems.''',
           add_help=False)
parser.add_argument('-aggressive', action='store_true', default=False,
                    help='for extruded hierarchy, refine aggressively in the vertical (factor 4 instead of 2)')
parser.add_argument('-big', action='store_true', default=False,
                    help='multiply all domain dimensions by 10^5 = 100.0e3; should make no difference')
parser.add_argument('-mx', type=int, default=1, metavar='N',
                    help='for coarse/base mesh, number of equal subintervals in x-direction (default=1)')
parser.add_argument('-my', type=int, default=1, metavar='N',
                    help='for coarse/base mesh, number of equal subintervals in y-direction (default=1)')
parser.add_argument('-mz', type=int, default=1, metavar='N',
                    help='for coarse/base mesh, number of element layers in each vertical column (default=1)')
parser.add_argument('-o', metavar='FILE.pvd', type=str, default='',
                    help='save results to .pvd file')
parser.add_argument('-solehelp', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-printparams', action='store_true', default=False,
                    help='print dictionary of solver parameters')
parser.add_argument('-refine', type=int, default=0, metavar='N',
                    help='refine all dimensions in stage 1, otherwise number of vertical (z) mesh refinements when extruding (default=0)')
parser.add_argument('-stage', type=int, default=1, metavar='S',
                    help='problem stage 1,2,3 (default=1)')
args, unknown = parser.parse_known_args()
if args.solehelp:
    parser.print_help()
    sys.exit(0)

if args.stage > 3:
    raise NotImplementedError('only stages 1,2,3 exist')
if args.stage == 1 and args.aggressive:
    raise NotImplementedError('aggressive vertical coarsening only in stages 2,3')

# mesh and geometry: stage > 1 use extruded mesh
if args.stage in {1,2}:
    L = 1.0
    H = 1.0
else:
    L = 1.0
    H = 0.01
if args.big:
    L *= 100.0e3
    H *= 100.0e3
if args.stage == 1:
    basecoarse = RectangleMesh(args.mx,args.my,L,L,quadrilateral=True)
    mx = args.mx * 2**args.refine
    my = args.my * 2**args.refine
    basehierarchy = MeshHierarchy(basecoarse,args.refine)
    mz = args.mz * 2**args.refine
    hierarchy = ExtrudedMeshHierarchy(basehierarchy,H,base_layer=args.mz,
                                      refinement_ratio=2)
else:
    mx = args.mx
    my = args.my
    base = RectangleMesh(mx,my,L,L,quadrilateral=True)
    rz = 4 if args.aggressive else 2
    mz = args.mz * rz**args.refine
    hierarchy = SemiCoarsenedExtrudedHierarchy(base,H,base_layer=args.mz,
                                               refinement_ratio=rz,nref=args.refine)
mesh = hierarchy[-1]
PETSc.Sys.Print('extruded mesh:      %d x %d x %d mesh of Q1 elements on %.2f x %.2f x %.2f domain' \
                 % (mx,my,mz,L,L,H))

# function space
V = FunctionSpace(mesh, 'Q', 1)
PETSc.Sys.Print('vector space dim:   N=%d' % V.dim())

# source function f
x, y, z = SpatialCoordinate(mesh)
uexact = Function(V).interpolate( sin(2.0*pi*x/L) * sin(2.0*pi*y/L) * sin(pi*z/(2.0*H)) )
f = Function(V).interpolate( pi**2 * ( 8.0/L**2 + 1.0/(4.0*H**2) ) * uexact )

# weak form
u = Function(V)
v = TestFunction(V)
F = (inner(grad(u), grad(v)) - inner(f, v))*dx

# boundary conditions: natural on top but otherwise zero Dirichlet all around
bcs = [DirichletBC(V, Constant(0), (1, 2, 3, 4)),
       DirichletBC(V, Constant(0), 'bottom')]

# solver parameters
params = {'snes_type': 'ksponly',
          'ksp_type': 'cg',
          'ksp_converged_reason': None,
          'pc_type': 'mg',
          'mg_coarse_ksp_type': 'preonly'}
if args.stage == 1:
    params['mg_coarse_pc_type'] = 'lu'
    params['mg_coarse_pc_factor_mat_solver_type'] = 'mumps'
else:
    params['mg_coarse_pc_type'] = 'gamg'

# note that the printed parameters *do not* include -s_xxx_yyy type overrides
if args.printparams:
    pprint(params)
mesh._topology_dm.viewFromOptions('-dm_view')  # shows DMPlex view for base mesh in stages 2,3

# set up solver and report MG structure
problem = NonlinearVariationalProblem(F, u, bcs=bcs)
solver = NonlinearVariationalSolver(problem,solver_parameters=params,options_prefix='s')

# solve
solver.solve()

# report on GMG and AMG levels; the latter are only known *after* solve (i.e. PCSetup)
pc = solver.snes.ksp.pc
coarsepc = pc.getMGCoarseSolve().pc
if args.stage == 1:
    assert(coarsepc.getMGLevels() == 0)
    PETSc.Sys.Print('  3D coarsening:    GMG levels = %d' % pc.getMGLevels())
else:
    PETSc.Sys.Print('  semi-coarsening:  GMG levels = %d, coarse-level AMG levels = %d' \
                    % (pc.getMGLevels(),coarsepc.getMGLevels()))

# report numerical error
L2err = sqrt(assemble(dot(u - uexact, u - uexact) * dx))
PETSc.Sys.Print('  L_2 error norm = %g' % L2err)

# optionally save to file
if args.o:
    u.rename('solution')
    if mesh.comm.size > 1:
        rank = Function(FunctionSpace(mesh,'DG',0))
        rank.dat.data[:] = mesh.comm.rank  # element-wise process rank
        rank.rename('rank')
        PETSc.Sys.Print('writing solution u and rank to %s ...' % args.o)
        File(args.o).write(u,rank)
    else:
        PETSc.Sys.Print('writing solution u to %s ...' % args.o)
        File(args.o).write(u)

