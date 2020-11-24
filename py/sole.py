#!/usr/bin/env python3

# notes: compare the following solvers, all on 16x16x16 meshes of Q1 hexs or Q1 prisms(x2):
#   ./sole.py -refine 4 -stage 1
#   ./sole.py -mx 16 -my 16 -refine 4 -stage 2
#   ./sole.py -mx 16 -my 16 -refine 2 -stage 2 -aggressive
#   ./sole.py -mx 16 -my 16 -refine 2 -stage 2 -aggressive -prism
#   ./sole.py -mx 16 -my 16 -refine 4 -stage 3
#   ./sole.py -mx 16 -my 16 -refine 2 -stage 3 -aggressive
#   ./sole.py -mx 16 -my 16 -refine 2 -stage 3 -aggressive -prism

# results
# on 128x128x64 meshes with N=10^6 we get higher performance from aggressive
# semi-coarsening, especially in the high aspect (thin layer) case, but no
# performance benefit from using prisms
#   $ tmpg -n 1 ./sole.py -mx 2 -my 2 -refine 6 -stage 1        # for reference: 3D GMG refinement
#     Linear s_ solve converged due to CONVERGED_RTOL iterations 2
#     3D coarsening:    GMG levels = 7
#     L_2 error norm = 6.93709e-05
#   real 65.00
#   $ tmpg -n 1 ./sole.py -mx 128 -my 128 -refine 6 -stage 2
#     Linear s_ solve converged due to CONVERGED_RTOL iterations 6
#     semi-coarsening:  GMG levels = 7, coarse-level AMG levels = 4
#     L_2 error norm = 6.95419e-05
#   real 111.86
#   $ tmpg -n 1 ./sole.py -mx 128 -my 128 -refine 3 -stage 2 -aggressive
#     Linear s_ solve converged due to CONVERGED_RTOL iterations 8
#     semi-coarsening:  GMG levels = 4, coarse-level AMG levels = 4
#     L_2 error norm = 6.95734e-05
#   real 77.90                                                  # 20% loss over 3D GMG
#   $ tmpg -n 1 ./sole.py -mx 128 -my 128 -refine 3 -stage 2 -aggressive -prism
#     Linear s_ solve converged due to CONVERGED_RTOL iterations 12
#     semi-coarsening:  GMG levels = 4, coarse-level AMG levels = 4
#     L_2 error norm = 0.000235012
#   real 82.44                                                  # 27% loss over 3D GMG
#   $ tmpg -n 1 ./sole.py -mx 128 -my 128 -refine 6 -stage 3
#     Linear s_ solve converged due to CONVERGED_RTOL iterations 3
#     semi-coarsening:  GMG levels = 7, coarse-level AMG levels = 4
#     L_2 error norm = 1.79121e-06
#   real 106.58
#   $ tmpg -n 1 ./sole.py -mx 128 -my 128 -refine 3 -stage 3 -aggressive
#     Linear s_ solve converged due to CONVERGED_RTOL iterations 3
#     semi-coarsening:  GMG levels = 4, coarse-level AMG levels = 4
#     L_2 error norm = 1.80111e-06
#   real 72.07                                                  # 11% loss over 3D GMG
#   $ tmpg -n 1 ./sole.py -mx 128 -my 128 -refine 3 -stage 3 -aggressive -prism
#     Linear s_ solve converged due to CONVERGED_RTOL iterations 3
#     semi-coarsening:  GMG levels = 4, coarse-level AMG levels = 4
#     L_2 error norm = 1.84718e-06
#   real 77.78                                                  # 20% loss over 3D GMG

# results
# high performance on a ice-sheet relevant grid aspect; runtime about 3 minutes
# for N=17.1e6 degrees of freedom; achieved here without matrix-free
#   $ tmpg -n 8 ./sole.py -mx 512 -my 512 -refine 3 -stage 3 -aggressive
#   stage 3:            extruded semicoarsening in z, high aspect, GMG in z and AMG in base
#   extruded mesh:      512 x 512 x 64 mesh of Q1 hex elements on 1.00 x 1.00 x 0.01 domain
#   vector space dim:   N=17105985
#     Linear s_ solve converged due to CONVERGED_RTOL iterations 4
#     semi-coarsening:  GMG levels = 4, coarse-level AMG levels = 5
#     L_2 error norm = 1.77313e-06
#   real 202.83


from firedrake import *
import sys, argparse
from pprint import pprint

parser = argparse.ArgumentParser(description='''
Three stages of multigrid Poisson solvers in 3D using regular meshes.
Set the coarsest grid with -mx, -my, -mz; defaults are
(mx,my,mz)=(1,1,1) and the default -refine is 0.  For stage 1 -refine acts
equally in all dimensions and the solution is ordinary GMG.  For stages 2,3
the GMG refinement is only in z (semi-coarsening) and the coarse grid problem
is solved by AMG (-mg_coarse_pc_type gamg).  In stages 2,3 the semi-coarsening
uses a default factor of 2 but -aggressive changes to 4.  Stages 1 and 2 are
on unit cubes (though stage 2 is managed as an extruded mesh).  Stage 3 is a
high-aspect ratio box with small height, again managed as an extruded mesh.
(Note that a sole is a kind of flat fish.)  The first three stages in pool.py
are analogs of the stages here, but for Stokes problems.''',
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
parser.add_argument('-prism', action='store_true', default=False,
                    help='use triangles in base mesh for extruded prism elements')
parser.add_argument('-refine', type=int, default=0, metavar='N',
                    help='stage 1: refine all dimensions; stages 2,3: z refinements (default=0)')
parser.add_argument('-stage', type=int, default=1, metavar='S',
                    help='problem stage 1,2,3 (default=1)')
args, unknown = parser.parse_known_args()
if args.solehelp:
    parser.print_help()
    sys.exit(0)

# report stage
if args.stage > 3:
    raise NotImplementedError('only stages 1,2,3 exist')
if args.stage == 1 and args.aggressive:
    raise NotImplementedError('aggressive vertical coarsening only in stages 2,3')
if args.stage == 1 and args.prism:
    raise NotImplementedError('prism elements only possible in stages 2,3')
stagedict = {1: 'equal refinement in 3D, unit aspect, all GMG',
             2: 'extruded semicoarsening in z, unit aspect, GMG in z and AMG in base',
             3: 'extruded semicoarsening in z, high aspect, GMG in z and AMG in base'}
PETSc.Sys.Print('stage %d:            %s' % (args.stage,stagedict[args.stage]))

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
    PETSc.Sys.Print('mesh:               %d x %d x %d mesh of Q1 hex elements on %.2f x %.2f x %.2f domain' \
                     % (mx,my,mz,L,L,H))
else:
    mx = args.mx
    my = args.my
    base = RectangleMesh(mx,my,L,L,quadrilateral=(not args.prism))
    rz = 4 if args.aggressive else 2
    mz = args.mz * rz**args.refine
    hierarchy = SemiCoarsenedExtrudedHierarchy(base,H,base_layer=args.mz,
                                               refinement_ratio=rz,nref=args.refine)
    PETSc.Sys.Print('extruded mesh:      %d x %d x %d mesh of Q1 %s elements on %.2f x %.2f x %.2f domain' \
                     % (mx,my,mz,'prism(x2)' if args.prism else 'hex',L,L,H))
mesh = hierarchy[-1]

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

