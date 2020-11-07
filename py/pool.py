#!/usr/bin/env python3

# TODO:
#   * implement stage 5

from firedrake import *
import sys, argparse
from pprint import pprint

parser = argparse.ArgumentParser(description='''
Five stages of Stokes solvers in fixed 3D domains, so that performance
degradation can be assessed as we build toward a real ice sheet.  (Compare
stokesi.py for a real case, and sole.py for the easier Poisson problem.)
Starting point (stage 1) is linear Stokes with lid-driven Dirichlet boundary
conditions on a unit cube.  Stage 5 is regularized Glen-Stokes physics with a
stress-free surface and hilly topography on a high aspect ratio (100-to-1)
domain with ice sheet realistic dimensions.  All stages have nonslip conditions
on base and sides, i.e. these are swimming pools of fluid/ice.  All use Q2xQ1
mixed elements on hexahedra, and all solvers are based on Schur fieldsplit
solvers with GMG for the u-u block.  In stages > 1 cases the GMG is only via
vertical semi-coarsening, and the coarse mesh is solved by AMG
(-mg_coarse_pc_type gamg).  At each stage the best solver, among the options
tested, is identified.

Stages:
    1. linear Stokes, flat top (w/o gravity), lid-driven top, unit cube, 3D GMG
    2. linear Stokes, flat top (w/o gravity), lid-driven top, unit cube, *
    3. linear Stokes, topography, stress-free surface, unit cube, *
    4. linear Stokes, topography, stress-free surface, high-aspect geometry, *
    5. Glen-law Stokes, topography, stress-free surface, high-aspect geometry, *
For stages 2-5:
    * = GMG vertical semicoarsening with AMG on the 2D base mesh.

Set the coarsest grid with -mx, -my, -mz; defaults are (mx,my,mz)=(1,1,1) and
the default -refine is 0.  For stage 1 -refine acts equally in all dimensions
(and GMG acts equally).  In stages 2-5 the semi-coarsening uses the default
refinement ratio of 2 but with -aggressive the factor is 4.
''',
           add_help=False)
parser.add_argument('-aggressive', action='store_true', default=False,
                    help='refine by 4 in vertical semicoarsening (instead of 2)')
parser.add_argument('-mx', type=int, default=1, metavar='N',
                    help='for coarse/base mesh, number of equal subintervals in x-direction (default=1)')
parser.add_argument('-my', type=int, default=1, metavar='N',
                    help='for coarse/base mesh, number of equal subintervals in y-direction (default=1)')
parser.add_argument('-mz', type=int, default=1, metavar='N',
                    help='for coarse/base mesh, number of element layers in each vertical column (default=1)')
parser.add_argument('-o', metavar='FILE.pvd', type=str, default='',
                    help='save results to .pvd file')
parser.add_argument('-poolhelp', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-printparams', action='store_true', default=False,
                    help='print dictionary of solver parameters')
parser.add_argument('-refine', type=int, default=0, metavar='N',
                    help='refine all dimensions in stage 1, otherwise number of vertical (z) mesh refinements when extruding (default=0)')
parser.add_argument('-stage', type=int, default=1, metavar='S',
                    help='problem stage 1,...,6 (default=1)')
args, unknown = parser.parse_known_args()
if args.poolhelp:
    parser.print_help()
    sys.exit(0)

if args.stage > 3:
    raise NotImplementedError('only stages 1,2,3 so far')
if args.stage == 1 and args.aggressive:
    raise NotImplementedError('aggressive vertical coarsening only in stages > 1')

# geometry
if args.stage > 3:
    L = 100.0e3
    H = 1000.0
else:
    L = 1.0
    H = 1.0

# mesh:  fine mesh is mx x my x mz;  use base mesh hierarchy only in stage 1
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
    rz = 4 if args.aggressive else 2   # vertical refinement ratio
    mz = args.mz * rz**args.refine
    hierarchy = SemiCoarsenedExtrudedHierarchy(base,H,base_layer=args.mz,
                                               refinement_ratio=rz,nref=args.refine)
mesh = hierarchy[-1]
PETSc.Sys.Print('extruded mesh:      %d x %d x %d hex mesh on %.2f x %.2f x %.2f domain' \
                 % (mx,my,mz,L,L,H))

if args.stage > 3: #FIXME
    raise NotImplementedError('only stages 1,2,3 so far')

# deform mesh coordinates
if args.stage > 2:
    for kmesh in hierarchy:
        Vcoord = kmesh.coordinates.function_space()
        x,y,z = SpatialCoordinate(kmesh)
        h = 1.0 + sin(3.0*pi*x) * sin(2.0*pi*y)  # FIXME suitable only for 1,2,3
        f = Function(Vcoord).interpolate(as_vector([x,y,h*z]))
        kmesh.coordinates.assign(f)

# function spaces
V = VectorFunctionSpace(mesh, 'Q', 2)
W = FunctionSpace(mesh, 'Q', 1)
Z = V * W
n_u,n_p,N = V.dim(),W.dim(),Z.dim()
PETSc.Sys.Print('vector space dims:  n_u=%d, n_p=%d  -->  N=%d' % (n_u,n_p,N))

FIXME add g body force

# weak form
up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)
# symmetric gradient & divergence terms in F
F = (inner(grad(u), grad(v)) - p * div(v) - div(u) * q - inner(Constant((0, 0, 0)), v))*dx

# some methods may use a mass matrix for preconditioning the Schur block
class Mass(AuxiliaryOperatorPC):

    def form(self, pc, test, trial):
        a = inner(test, trial)*dx
        bcs = None
        return (a, bcs)

FIXME add stress-free surface as appropriate

# boundary conditions
# (for stages 1,2,3: velocity on lid is (1,1) equal in both x and y directions
if args.stage == 1:
    bcs = [DirichletBC(Z.sub(0), Constant((1.0, 1.0, 0)), 6),
           DirichletBC(Z.sub(0), Constant((0, 0, 0)), (1, 2, 3, 4, 5))]
else:
    bcs = [DirichletBC(Z.sub(0), Constant((1.0, 1.0, 0)), 'top'),
           DirichletBC(Z.sub(0), Constant((0, 0, 0)), (1, 2, 3, 4)),
           DirichletBC(Z.sub(0), Constant((0, 0, 0)), 'bottom')]

FIXME nullspace only when lid-driven

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

params = {'snes_type': 'ksponly',
          'ksp_type':  'gmres',    # on -stage 1, fgmres adds 10% to iterations and 3% to time
                                   #     and gcr acts the same as fgmres
          'ksp_converged_reason': None,
          'pc_type': 'fieldsplit',
          'pc_fieldsplit_type': 'schur',
          'pc_fieldsplit_schur_fact_type': 'lower',
          'pc_fieldsplit_schur_precondition': 'selfp'}
          #'pc_fieldsplit_schur_precondition': 'a11',
          #'pc_fieldsplit_schur_scale': -1.0,  # only active for diag
          #'fieldsplit_1_pc_type': 'python',
          #'fieldsplit_1_pc_python_type': '__main__.Mass',
          #'fieldsplit_1_aux_pc_type': 'bjacobi',
          #'fieldsplit_1_aux_sub_pc_type': 'icc'}

# turn on Newton and counting if nonlinear
if args.stage >= 4:
    params['snes_type'] = 'newtonls'
    params['snes_converged_reason'] = None

# only -stage 1 CAN use nest, because aij is required for extruded meshes,
#     but it is hard to argue from profiling that it makes any difference
#     even for -stage 1
if args.stage == 1:
    params['mat_type'] = 'nest'
else:
    params['mat_type'] = 'aij'

params['fieldsplit_0_ksp_type'] = 'preonly'
params['fieldsplit_0_pc_type'] = 'mg'
# the next three settings are actually faster (about 10%) than the default
#     (i.e. chebyshev,sor) on -stage 1, and FIXME PRESUMABLY more capable in high aspect ratio
params['fieldsplit_0_mg_levels_ksp_type'] = 'richardson'
params['fieldsplit_0_mg_levels_pc_type'] = 'ilu'
params['fieldsplit_0_mg_coarse_ksp_type'] = 'preonly'

if args.stage == 1:
    params['fieldsplit_0_mg_coarse_pc_type'] = 'lu'
    params['fieldsplit_0_mg_coarse_pc_factor_mat_solver_type'] = 'mumps'
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

# note that the printed parameters *do not* include -s_xxx_yyy type overrides
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

