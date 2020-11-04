#!/usr/bin/env python3

# TODO:
#   * implement stages 4,5,6

from firedrake import *
import sys, argparse
from pprint import pprint

parser = argparse.ArgumentParser(description='''
Six stages of Stokes in 3D domains with fixed geometry, so that performance
loss can be assessed.  Starting point is linear Stokes with lid-driven
Dirichlet boundary conditions on a unit cube.  Final destination is
regularized Glen-Stokes physics with a stress-free surface and hilly topography
on a high aspect ratio (100-to-1) domain with glacier-realistic dimensions.
All stages have nonslip conditions on base and sides, i.e. these are swimming
pools, and use Q2xQ1 mixed elements on hexahedra.  At each stage the best
solver, among the options tested of course, is identified.''',
           add_help=False)
parser.add_argument('-mx', type=int, default=2, metavar='N',
                    help='number of equal subintervals in x-direction (default=2)')
parser.add_argument('-my', type=int, default=2, metavar='N',
                    help='number of equal subintervals in y-direction (default=2)')
parser.add_argument('-mz', type=int, default=2, metavar='N',
                    help='number of layers in each vertical column (default=2)')
parser.add_argument('-o', metavar='FILE.pvd', type=str, default='',
                    help='save mesh and solution (u,p) to .pvd file')
parser.add_argument('-poolhelp', action='store_true', default=False,
                    help='print help for stokes2D.py and quit')
parser.add_argument('-printparams', action='store_true', default=False,
                    help='print dictionary of solver parameters')
parser.add_argument('-refine', type=int, default=0, metavar='N',
                    help='number of vertical (z) mesh refinements (default=0)')
parser.add_argument('-stage', type=int, default=1, metavar='S',
                    help='problem stage 1,...,6 (default=1)')
args, unknown = parser.parse_known_args()
if args.poolhelp:
    parser.print_help()
    sys.exit(0)

if args.stage > 3:
    raise NotImplementedError('only stages 1,2,3 so far')

# mesh and geometry: stage > 1 use extruded mesh
if args.stage in {1,2}:
    L = 1.0
    H = 1.0
else:
    L = 100.0e3
    H = 1000.0
mx = args.mx * 2**args.refine
my = args.my * 2**args.refine
mz = args.mz * 2**args.refine
if args.stage == 1:
    mesh = UnitCubeMesh(args.mx, args.my, args.mz)
    hierarchy = MeshHierarchy(mesh, args.refine)
else:
    base = RectangleMesh(mx,my,L,L,quadrilateral=True)
    hierarchy = SemiCoarsenedExtrudedHierarchy(base,H,base_layer=args.mz,nref=args.refine)
mesh = hierarchy[-1]
PETSc.Sys.Print('mesh:               %d x %d x %d mesh on %.2f x %.2f x %.2f domain' \
                % (mx,my,mz,L,L,H))

# function spaces
if args.stage == 1:
    V = VectorFunctionSpace(mesh, 'P', 2)
    W = FunctionSpace(mesh, 'P', 1)
else:
    V = VectorFunctionSpace(mesh, 'Q', 2)
    W = FunctionSpace(mesh, 'Q', 1)
Z = V * W
n_u,n_p,N = V.dim(),W.dim(),Z.dim()
PETSc.Sys.Print('vector space dims:  n_u=%d, n_p=%d  -->  N=%d' % (n_u,n_p,N))

# weak form
up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)
# symmetric gradient & divergence terms in F
F = (inner(grad(u), grad(v)) - p * div(v) - div(u) * q - inner(Constant((0, 0, 0)), v))*dx

# boundary conditions
# (for stages 1,2,3: velocity on lid is (1,1) equal in both x and y directions
if args.stage == 1:
    bcs = [DirichletBC(Z.sub(0), Constant((1.0, 1.0, 0)), 6),
           DirichletBC(Z.sub(0), Constant((0, 0, 0)), (1, 2, 3, 4, 5))]
else:
    bcs = [DirichletBC(Z.sub(0), Constant((1.0, 1.0, 0)), 'top'),
           DirichletBC(Z.sub(0), Constant((0, 0, 0)), (1, 2, 3, 4)),
           DirichletBC(Z.sub(0), Constant((0, 0, 0)), 'bottom')]

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

params = {'snes_type': 'ksponly',
          'ksp_type':  'gmres',
          'pc_type': 'fieldsplit',
          'pc_fieldsplit_type': 'schur',
          'pc_fieldsplit_schur_fact_type': 'lower',
          'pc_fieldsplit_schur_precondition': 'selfp'}

if args.stage == 1:
    params['mat_type'] = 'nest'
else:
    params['mat_type'] = 'aij'

params['fieldsplit_0_ksp_type'] = 'preonly'
params['fieldsplit_0_pc_type'] = 'mg'
params['fieldsplit_0_mg_coarse_ksp_type'] = 'preonly'

if args.stage == 1:
    params['fieldsplit_0_mg_coarse_pc_type'] = 'lu'
    parameters['fieldsplit_0_mg_coarse_pc_factor_mat_solver_type'] = 'mumps'
else:
    params['fieldsplit_0_mg_coarse_pc_type'] = 'gamg'

params['fieldsplit_1_ksp_type'] = 'preonly'
params['fieldsplit_1_pc_type'] = 'jacobi'
params['fieldsplit_1_pc_jacobi_type'] = 'diagonal'

#params['fieldsplit_1_ksp_type'] = 'richardson'
#params['fieldsplit_1_ksp_max_it'] = 3
#params['fieldsplit_1_ksp_converged_maxits'] = None
#params['fieldsplit_1_ksp_richardson_scale'] = 1.0
#params['fieldsplit_1_ksp_type'] = 'gmres'
#params['fieldsplit_1_ksp_rtol'] = 0.1
#params['fieldsplit_1_pc_type'] = 'bjacobi'
#params['fieldsplit_1_sub_pc_type'] = 'ilu'

if args.printparams:
    pprint(params)

solve(F == 0, up, bcs=bcs, nullspace=nullspace, solver_parameters=params,
      options_prefix='s')

if args.o:
    u, p = up.split()
    u.rename('velocity')
    p.rename('pressure')
    if mesh.comm.size > 1:
        rank = Function(FunctionSpace(mesh,'DG',0))
        rank.dat.data[:] = mesh.comm.rank  # element-wise process rank
        rank.rename('rank')
        PETSc.Sys.Print('writing solution (u,p) and rank to %s ...' % args.o)
        File(args.o).write(u,p,rank)
    else:
        PETSc.Sys.Print('writing solution (u,p) to %s ...' % args.o)
        File(args.o).write(u,p)

