#!/usr/bin/env python3

from firedrake import *
import sys, argparse
from pprint import pprint

parser = argparse.ArgumentParser(description='''
Three stages of Poisson in 3D, analogs of the first three stages in pool.py.
Note that a sole is a kind of flat fish.''',
           add_help=False)
parser.add_argument('-mx', type=int, default=1, metavar='N',
                    help='number of equal subintervals in x-direction (default=1)')
parser.add_argument('-my', type=int, default=1, metavar='N',
                    help='number of equal subintervals in y-direction (default=1)')
parser.add_argument('-mz', type=int, default=1, metavar='N',
                    help='number of layers in each vertical column (default=1)')
parser.add_argument('-o', metavar='FILE.pvd', type=str, default='',
                    help='save mesh and solution to .pvd file')
parser.add_argument('-solehelp', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-printparams', action='store_true', default=False,
                    help='print dictionary of solver parameters')
parser.add_argument('-refine', type=int, default=1, metavar='N',
                    help='number of vertical (z) mesh refinements (default=1)')
parser.add_argument('-stage', type=int, default=1, metavar='S',
                    help='problem stage 1,2,3 (default=1)')
args, unknown = parser.parse_known_args()
if args.solehelp:
    parser.print_help()
    sys.exit(0)

if args.stage > 3:
    raise NotImplementedError('only stages 1,2,3 exist')

# mesh and geometry: stage > 1 use extruded mesh
if args.stage in {1,2}:
    L = 1.0
    H = 1.0
else:
    L = 1.0
    H = 0.1  # FIXME decrease to 0.01 once I get it
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

# function spaces: P1 or Q1
if args.stage == 1:
    V = FunctionSpace(mesh, 'P', 1)
    el = 'P1'
else:
    V = FunctionSpace(mesh, 'Q', 1)
    el = 'Q1'
PETSc.Sys.Print('mesh:               %d x %d x %d mesh of %s elements on %.2f x %.2f x %.2f domain' \
                % (mx,my,mz,el,L,L,H))
PETSc.Sys.Print('vector space dim:   N=%d' % V.dim())

# source function f
x, y, z = SpatialCoordinate(mesh)
uexact = Function(V).interpolate( sin(2.0*pi*x) * sin(2.0*pi*y) * sin(pi*z/(2.0*H)) )
f = Function(V).interpolate( pi**2 * (8.0 + (1.0/(2.0*H))**2) * uexact )

# weak form
u = Function(V)
v = TestFunction(V)
F = (inner(grad(u), grad(v)) - inner(f, v))*dx

# boundary conditions: natural on top but otherwise zero Dirichlet all around
if args.stage == 1:
    bcs = [DirichletBC(V, Constant(0), (1, 2, 3, 4, 5))]
else:
    bcs = [DirichletBC(V, Constant(0), (1, 2, 3, 4)),
           DirichletBC(V, Constant(0), 'bottom')]

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

# solve and report numerical error
solve(F == 0, u, bcs=bcs, solver_parameters=params,
      options_prefix='s')
L2err = sqrt(assemble(dot(u - uexact, u - uexact) * dx))
PETSc.Sys.Print('L_2 error norm = %g' % L2err)

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

