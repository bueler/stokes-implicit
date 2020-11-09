#!/usr/bin/env python3

# TODO:
#   * check this is fixed: diagnose why   ./pool.py -stage 3 -topomag 0.0   is wrong solution (on various meshes)
#   * optimality scripts for stage 1 (below), stage 2 (below), stage 3 (TODO), stage 4 (TODO)
#   * implement stage 5

# evidence of stage 1 optimality on 4^3,8^3,16^3,32^3,64^3 meshes; note KSPSolve is only 63% of time on last grid
#$ for LEV in 0 1 2 3 4; do tmpg -n 1 ./pool.py -stage 1 -mx 4 -my 4 -mz 4 -refine $LEV -log_view &> lev$LEV.txt; 'grep' "KSPSolve " lev$LEV.txt; 'grep' "solve converged" lev$LEV.txt; done
#KSPSolve               1 1.0 1.7344e-01 1.0 8.45e+07 1.0 0.0e+00 0.0e+00 0.0e+00 11100  ...
#  Linear s_ solve converged due to CONVERGED_RTOL iterations 23
#KSPSolve               1 1.0 3.6765e+00 1.0 1.41e+10 1.0 0.0e+00 0.0e+00 0.0e+00 65100  ...
#  Linear s_ solve converged due to CONVERGED_RTOL iterations 18
#KSPSolve               1 1.0 1.4665e+01 1.0 2.40e+10 1.0 0.0e+00 0.0e+00 0.0e+00 68100  ...
#  Linear s_ solve converged due to CONVERGED_RTOL iterations 20
#KSPSolve               1 1.0 1.0338e+02 1.0 9.43e+10 1.0 0.0e+00 0.0e+00 0.0e+00 66100  ...
#  Linear s_ solve converged due to CONVERGED_RTOL iterations 21
#KSPSolve               1 1.0 8.0955e+02 1.0 6.45e+11 1.0 0.0e+00 0.0e+00 0.0e+00 63100  .
#  Linear s_ solve converged due to CONVERGED_RTOL iterations 21

# for stage 2, measure these runs on 4^3,8^3,16^3,32^3,64^3 meshes:
#   $ tmpg -n 1 ./pool.py -stage 2 -mx 4 -my 4 -refine 1 -aggressive
#   $ tmpg -n 1 ./pool.py -stage 2 -mx 8 -my 8 -mz 2 -refine 1 -aggressive
#   $ tmpg -n 1 ./pool.py -stage 2 -mx 16 -my 16 -refine 2 -aggressive
#   $ tmpg -n 1 ./pool.py -stage 2 -mx 32 -my 32 -mz 2 -refine 2 -aggressive
#   $ tmpg -n 1 ./pool.py -stage 2 -mx 64 -my 64 -refine 3 -aggressive

import sys, argparse
from pprint import pprint

from firedrake import *
#possibly: PETSc.Sys.popErrorHandler()

import src.constants as consts

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
parser.add_argument('-topomag', type=float, default=0.3, metavar='N',
                    help='for stages 3,4,5: the relative magnitude of surface topography (default=0.3)')
args, unknown = parser.parse_known_args()
if args.poolhelp:
    parser.print_help()
    sys.exit(0)

if args.stage > 4:
    raise NotImplementedError('only stages 1--4 so far')
if args.stage == 1 and args.aggressive:
    raise NotImplementedError('aggressive vertical coarsening only in stages > 1')

# geometry: L x L x H
if args.stage in {4,5}:
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
PETSc.Sys.Print('extruded mesh:      %d x %d x %d hexahedra' % (mx,my,mz))

# deform z coordinate of mesh
if args.stage in {3,4,5}:
    for kmesh in hierarchy:
        Vcoord = kmesh.coordinates.function_space()
        x,y,z = SpatialCoordinate(kmesh)
        h = H * (1.0 + args.topomag * sin(3.0*pi*x/L) * sin(2.0*pi*y/L))  # FIXME not o.k. for stage 5?
        f = Function(Vcoord).interpolate(as_vector([x,y,h*z]))
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
    Q1base = FunctionSpace(mesh._base_mesh,'Q',1)
    hfcn = Function(Q1base)
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
V = VectorFunctionSpace(mesh, 'Q', 2)
W = FunctionSpace(mesh, 'Q', 1)
Z = V * W
n_u,n_p,N = V.dim(),W.dim(),Z.dim()
PETSc.Sys.Print('vector space dims:  n_u=%d, n_p=%d  -->  N=%d' % (n_u,n_p,N))

# weak form
up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)
if args.stage == 1:
    f_body = Constant((0, 0, 0))
elif args.stage in {2,3}:
    f_body = Constant((0, 0, -consts.g))  # density = 1.0
elif args.stage in {4,5}:
    f_body = Constant((0, 0, -consts.rho*consts.g))
# note symmetric gradient & divergence terms in F
F = (inner(grad(u), grad(v)) - p * div(v) - div(u) * q - inner(f_body, v))*dx  # FIXME change for stage 5

## some methods may use a mass matrix for preconditioning the Schur block
#class Mass(AuxiliaryOperatorPC):
#
#    def form(self, pc, test, trial):
#        a = inner(test, trial)*dx  # FIXME use viscosity?
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

params = {'mat_type': 'aij',       # FIXME experiment with matfree
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

# these smoother settings may be more capable in high aspect ratio;
# they are essentially the same speed in stage 1
#params['fieldsplit_0_mg_levels_ksp_type'] = 'richardson'
#params['fieldsplit_0_mg_levels_pc_type'] = 'ilu'

if args.stage == 1:
    #pass
    # either parallel LU via MUMPS or serial LU on each process
    params['fieldsplit_0_mg_coarse_pc_type'] = 'lu'
    params['fieldsplit_0_mg_coarse_pc_factor_mat_solver_type'] = 'mumps'
    #params['fieldsplit_0_mg_coarse_pc_type'] = 'redundant'
    #params['fieldsplit_0_mg_coarse_sub_pc_type'] = 'lu'
    #params['fieldsplit_0_mg_coarse_pc_type'] = 'ilu'
else:
    params['fieldsplit_0_mg_coarse_pc_type'] = 'gamg'

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
if args.printparams:
    pprint(params)

# set up solver and report MG structure
problem = NonlinearVariationalProblem(F, up, bcs=bcs)
solver = NonlinearVariationalSolver(problem,
                                    nullspace=nullspace,
                                    solver_parameters=params,
                                    options_prefix='s')

# solve
solver.solve()

## report on GMG and AMG levels; the latter are only known *after* solve (i.e. PCSetup)
pc = solver.snes.ksp.pc
assert(pc.getType() == 'fieldsplit')
pc0 = pc.getFieldSplitSubKSP()[0].pc
assert(pc0.getType() == 'mg')
coarsepc0 = pc0.getMGCoarseSolve().pc
if args.stage == 1:
    assert(coarsepc0.getMGLevels() == 0)
    PETSc.Sys.Print('  3D coarsening:    GMG levels = %d' % pc0.getMGLevels())
else:
    PETSc.Sys.Print('  semi-coarsening:  GMG levels = %d, coarse-level AMG levels = %d' \
                    % (pc0.getMGLevels(),coarsepc0.getMGLevels()))

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

