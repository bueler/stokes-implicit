#!/usr/bin/env python3

# TODO:
#   * rethink scaling in stage 4,5 problems
#   * optimality scripts for stage 4 (below)
#   * implement stage 5

# for stage 4, measure these runs on 8x8x1, 16x16x2, 32x32x4, 64x64x8, 128x128x16 meshes:
#   $ tmpg -n 8 ./pool.py -stage 4 -mx 8 -my 8 -refine 0
#   $ tmpg -n 8 ./pool.py -stage 4 -mx 16 -my 16 -mz 2 -refine 0
#   $ tmpg -n 8 ./pool.py -stage 4 -mx 32 -my 32 -refine 1 -aggressive
#   $ tmpg -n 8 ./pool.py -stage 4 -mx 64 -my 64 -mz 2 -refine 1 -aggressive
#   $ tmpg -n 8 ./pool.py -stage 4 -mx 128 -my 128 -refine 2 -aggressive

import sys, argparse
from pprint import pprint

from firedrake import *
#possibly: PETSc.Sys.popErrorHandler()

import src.constants as consts

parser = argparse.ArgumentParser(description='''
Five stages of Stokes solvers in fixed 3D domains, so that performance
degradation can be assessed as we build toward a real ice sheet.  (Compare
stokesi.py for a real case, and sole.py for the easier Poisson problem.)  The
starting point (stage 1) is linear Stokes with lid-driven Dirichlet boundary
conditions on a unit cube.  Stage 5 is regularized Glen-Stokes physics with a
stress-free surface and hilly topography on a high aspect ratio (100-to-1)
domain with ice sheet dimensions.  All stages have nonslip conditions on their
base and sides, i.e. these are swimming pools of fluid/ice.

In all stages the FEM uses a base mesh of triangles, extrudes the mesh to
prisms, and applies Q2xDP0 mixed elements.  The solvers are all based
on Schur fieldsplit with GMG for the u-u block.  Stage 1 has a standard 3D
GMG solver for the u-u block.  In stages 2-5 the GMG
is only via vertical semi-coarsening, and the coarse mesh is solved by AMG
(-mg_coarse_pc_type gamg).  At each stage the best solver, among the options
tested, is identified.

Stages:
    1. linear Stokes, flat top (w/o gravity), lid-driven top, unit cube, 3D GMG
    2. linear Stokes, flat top, lid-driven top, unit cube, *
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
parser.add_argument('-topomag', type=float, default=0.5, metavar='N',
                    help='for stages 3,4,5: relative magnitude of surface topography (default=0.5)')
args, unknown = parser.parse_known_args()
if args.poolhelp:
    parser.print_help()
    sys.exit(0)

# report stage
if args.stage > 4:
    raise NotImplementedError('only stages 1--4 so far')
if args.stage == 1 and args.aggressive:
    raise NotImplementedError('aggressive vertical coarsening only in stages > 1')
stagedict = {1: 'lid-driven unit-cube cavity, 3D GMG',
             2: 'lid-driven unit-cube cavity, GMG in z only',
             3: 'topography on top of unit-cube, GMG in z only',
             4: 'topography on top of high-aspect, GMG in z only'}
PETSc.Sys.Print('stage %d:            %s' % (args.stage,stagedict[args.stage]))

# geometry: L x L x H
if args.stage in {4,5}:
    L = 100.0e3
    H = 1000.0
else:
    L = 1.0
    H = 1.0

# mesh
if args.stage == 1:
    basecoarse = RectangleMesh(args.mx,args.my,L,L)  # triangles
    mx = args.mx * 2**args.refine
    my = args.my * 2**args.refine
    basehierarchy = MeshHierarchy(basecoarse,args.refine)
    mz = args.mz * 2**args.refine
    hierarchy = ExtrudedMeshHierarchy(basehierarchy,H,base_layer=args.mz,
                                      refinement_ratio=2)
else:
    mx = args.mx
    my = args.my
    base = RectangleMesh(mx,my,L,L)    # triangles
    rz = 4 if args.aggressive else 2   # vertical refinement ratio
    mz = args.mz * rz**args.refine
    hierarchy = SemiCoarsenedExtrudedHierarchy(base,H,base_layer=args.mz,
                                               refinement_ratio=rz,nref=args.refine)
mesh = hierarchy[-1]
PETSc.Sys.Print('extruded mesh:      %d x %d x %d prisms(x2)' % (mx,my,mz))

# if top has topography then deform z coordinate of mesh
if args.stage in {3,4,5}:
    for kmesh in hierarchy:
        Vcoord = kmesh.coordinates.function_space()
        x,y,z = SpatialCoordinate(kmesh)
        hmult = 1.0 + args.topomag * sin(3.0*pi*x/L) * sin(2.0*pi*y/L)  # FIXME not o.k. for stage 5?
        f = Function(Vcoord).interpolate(as_vector([x,y,hmult*z]))
        kmesh.coordinates.assign(f)

# mesh coordinates
x,y,z = SpatialCoordinate(mesh)

# report on geometry
if args.stage in {1,2}:
    PETSc.Sys.Print('domain:             %.2f x %.2f x %.2f' % (L,L,H))
else:
    # get surface elevation back from fine mesh
    Q1 = FunctionSpace(mesh,'Q',1)
    hbc = DirichletBC(Q1,1.0,'top')  # we use hbc.nodes below
    P1base = FunctionSpace(mesh._base_mesh,'P',1)
    hfcn = Function(P1base)
    # z itself is an 'Indexed' object, so use a Function with a .dat attribute
    zfcn = Function(Q1).interpolate(z)
    # add halos for parallelizability of the interpolation
    hfcn.dat.data_with_halos[:] = zfcn.dat.data_with_halos[hbc.nodes]
    with hfcn.dat.vec_ro as vhfcn:
        zmin = vhfcn.min()[1]
        zmax = vhfcn.max()[1]
    PETSc.Sys.Print('domain:             %.2f x %.2f base domain with %.2f < z < %.2f at surface' \
                    % (L,L,zmin,zmax))

# function spaces
V = VectorFunctionSpace(mesh, 'Q', 2)   # Q2 on prisms
xpE = FiniteElement('DP',triangle,0)    # discontinuous P0 in horizontal ...
zpE = FiniteElement('P',interval,1)     #     tensored with continuous P1 in vertical
pE = TensorProductElement(xpE,zpE)
W = FunctionSpace(mesh, pE)

Z = V * W
n_u,n_p,N = V.dim(),W.dim(),Z.dim()
PETSc.Sys.Print('vector space dims:  n_u=%d, n_p=%d  -->  N=%d' % (n_u,n_p,N))

# body force is gravity for stage > 1
if args.stage == 1:
    f_body = Constant((0, 0, 0))
elif args.stage in {2,3}:
    f_body = Constant((0, 0, -consts.g))  # density = 1.0
elif args.stage in {4,5}:
    f_body = Constant((0, 0, -consts.rho*consts.g))

# viscosity
if args.stage in {1,2,3}:
    nu = 0.5
elif args.stage == 4:
    nu = 1.0e13  # corresponds to secpera = 31556926.0, A3 = 1.0e-16/secpera,
                 #                B3 = A3**(-1/3), Du = 6.2806e-09, nu = 0.5 * B3 * Du^(-2/3)
elif args.stage == 5:
    raise NotImplementedError  # FIXME change for stage 5

# weak form
up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)
# note symmetric gradient & divergence terms in F
if args.stage in {1,2,3,4}:
    F = (2.0 * nu * inner(grad(u), grad(v)) - p * div(v) - div(u) * q - inner(f_body, v))*dx
elif args.stage == 5:
    raise NotImplementedError  # FIXME change for stage 5

## some methods may use a mass matrix for preconditioning the Schur block
## follow: https://www.firedrakeproject.org/_modules/firedrake/preconditioners/massinv.html#MassInvPC
#class Mass(AuxiliaryOperatorPC):
#
#    def form(self, pc, test, trial):
#        a = inner((1.0/nu) * test, trial)*dx
#        bcs = None
#        return (a, bcs)

# boundary conditions:  normally stress-free surface
bcs = [DirichletBC(Z.sub(0), Constant((0, 0, 0)), (1, 2, 3, 4)),
       DirichletBC(Z.sub(0), Constant((0, 0, 0)), 'bottom')]
nullspace = None
if args.stage in {1,2}:
    u_lid = Function(V).interpolate(as_vector([4.0 * x * (1.0 - x),0.0,0.0]))
    bcs.append(DirichletBC(Z.sub(0), u_lid, 'top'))
    ## set nullspace to constant pressure fields
    nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

params = {'mat_type': 'aij',       # matfree does not work with Schur fieldsplit selfp (which assembles Bt A B),
                                   # but see https://www.firedrakeproject.org/demos/stokes.py.html
          'ksp_type':  'gmres',    # OLD INFO: fgmres adds 10% to iterations and 3% to time
                                   #     and gcr acts the same as fgmres
          'ksp_converged_reason': None,
          'pc_type': 'fieldsplit',
          'pc_fieldsplit_type': 'schur',
          'pc_fieldsplit_schur_fact_type': 'lower',  # 'upper' worse (why?); 'diag' much worse; 'full' worse
          'fieldsplit_0_ksp_type': 'preonly',
          'fieldsplit_0_pc_type': 'mg',
          'fieldsplit_0_mg_coarse_ksp_type': 'preonly'}

# turn on Newton and iteration counting if nonlinear
if args.stage == 5:
    params['snes_type'] = 'newtonls'
    params['snes_converged_reason'] = None
else:
    params['snes_type'] = 'ksponly'

# smoother which is more capable than Chebyshev + SSOR
# note default: -s_fieldsplit_0_mg_levels_ksp_max_it 2
params['fieldsplit_0_mg_levels_ksp_type'] = 'richardson'
params['fieldsplit_0_mg_levels_pc_type'] = 'bjacobi'
params['fieldsplit_0_mg_levels_sub_pc_type'] = 'ilu'

# parallel LU via MUMPS on coarse grid  (versus GAMG)
params['fieldsplit_0_mg_coarse_pc_type'] = 'lu'
params['fieldsplit_0_mg_coarse_pc_factor_mat_solver_type'] = 'mumps'

params['fieldsplit_1_ksp_type'] = 'preonly'

#params['fieldsplit_1_ksp_type'] = 'richardson'
#params['fieldsplit_1_ksp_max_it'] = 3
#params['fieldsplit_1_ksp_converged_maxits'] = None
#params['fieldsplit_1_ksp_richardson_scale'] = 1.0
#params['fieldsplit_1_ksp_type'] = 'gmres'
#params['fieldsplit_1_ksp_rtol'] = 0.1
#params['fieldsplit_1_pc_type'] = 'bjacobi'
#params['fieldsplit_1_sub_pc_type'] = 'ilu'

## from advice on PETSc man page for PCFieldSplitSetSchurPre
## but seems to actually give more iters and be slower than selp/jacobi strategy below
#params['pc_fieldsplit_schur_precondition'] = 'self'
#params['fieldsplit_1_pc_type'] = 'lsc'

params['pc_fieldsplit_schur_precondition'] = 'selfp'
params['fieldsplit_1_pc_type'] = 'jacobi'
params['fieldsplit_1_pc_jacobi_type'] = 'diagonal'

# WARNING: do not use following option which is faster than default 'diag' BUT CHANGES VELOCITY SOLUTION !?!?
#params['fieldsplit_1_mat_schur_complement_ainv_type'] = 'lump'

#params['pc_fieldsplit_schur_precondition'] = 'a11'
#params['fieldsplit_1_pc_type'] = 'python'
#params['fieldsplit_1_pc_python_type'] = '__main__.Mass'
##params['fieldsplit_1_aux_pc_type'] = 'jacobi'
#params['fieldsplit_1_aux_pc_type'] = 'bjacobi'
#params['fieldsplit_1_aux_sub_pc_type'] = 'icc'

#'pc_fieldsplit_schur_scale': -1.0,  # only active for diag

# note that the printed parameters *do not* include -s_xxx_yyy overrides
if args.printparams and mesh.comm.rank == 0:
    pprint(params)

# set up solver and report MG structure
problem = NonlinearVariationalProblem(F, up, bcs=bcs)
solver = NonlinearVariationalSolver(problem,
                                    nullspace=nullspace,
                                    solver_parameters=params,
                                    options_prefix='s')

# solve
solver.solve()

## report on GMG levels
pc = solver.snes.ksp.pc
assert(pc.getType() == 'fieldsplit')
pc0 = pc.getFieldSplitSubKSP()[0].pc
assert(pc0.getType() == 'mg')
if args.stage == 1:
    PETSc.Sys.Print('  3D coarsening:    GMG levels = %d' % pc0.getMGLevels())
else:
    PETSc.Sys.Print('  semi-coarsening:  GMG levels = %d' % pc0.getMGLevels())

# report solution norms
uL2 = sqrt(assemble(inner(u, u) * dx))
pL2 = sqrt(assemble(inner(p, p) * dx))
PETSc.Sys.Print('  solution norms:   |u|_2 = %.3e,  |p|_2 = %.3e' % (uL2,pL2))

# optionally save result
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

