#!/usr/bin/env python3

# compare matfreestokes.py
# see https://www.firedrakeproject.org/demos/stokes.py.html
# (see also https://www.firedrakeproject.org/matrix-free.html)

# results:  see study/mfmgstokes.sh and study/results/mfmgstokes.txt

import sys, argparse
from firedrake import *

parser = argparse.ArgumentParser(description='''
A matrix-free multigrid Stokes solver which is as simple as possible.
Problem is a 2D lid-driven cavity with stress-free base (thus no null
space).  Method is GMRES with Schur fieldsplit (lower) preconditioning.
All but the coarse level in the velocity block are matrix free.
The Schur block is preconditioned with ILU application of an assembled
mass matrix.
''',
           add_help=False)
parser.add_argument('-aggressive', action='store_true', default=False,
                    help='refine by 4 in vertical semicoarsening (instead of 2)')
parser.add_argument('-m0', type=int, default=16, metavar='N',
                    help='coarse grid is m0 x m0 mesh (default=16)')
parser.add_argument('-mfmghelp', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-o', metavar='FILE.pvd', type=str, default='',
                    help='save results to .pvd file')
parser.add_argument('-refine', type=int, default=1, metavar='N',
                    help='refine to generate multigrid hierarchy (default=1)')
args, unknown = parser.parse_known_args()
if args.mfmghelp:
    parser.print_help()
    sys.exit(0)

cmesh = UnitSquareMesh(args.m0, args.m0)
if args.aggressive:
    hierarchy = MeshHierarchy(cmesh, args.refine, refinements_per_level=2)
    nM = args.m0 * 4**args.refine
else:
    hierarchy = MeshHierarchy(cmesh, args.refine)
    nM = args.m0 * 2**args.refine
M = hierarchy[-1]     # the fine mesh
PETSc.Sys.Print('fine mesh:          %d x %d triangles(x2)' % (nM,nM))

V = VectorFunctionSpace(M, 'P', 2)
W = FunctionSpace(M, 'P', 1)
# alternative:  W = FunctionSpace(M, 'DP', 0)
Z = V * W
n_u,n_p,N = V.dim(),W.dim(),Z.dim()
PETSc.Sys.Print('vector space dims:  n_u=%d, n_p=%d  -->  N=%d' % (n_u,n_p,N))

up = Function(Z)
u,p = split(up)
v,q = TestFunctions(Z)

F = (inner(grad(u), grad(v)) - p * div(v) - div(u) * q)*dx
bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2))]

# preconditioner is GMRES+Schur(lower) with matfree GMG on the u-u block
#   smoother       = alpha=4/5 weighted Jacobi
#   coarse solver  = assembled LU
#   Schur block PC = mass matrix applied by ILU
parameters = {
    "mat_type": "matfree",
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "ksp_converged_reason": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "lower",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "mg",
    "fieldsplit_0_mg_levels_ksp_type": "richardson",
    "fieldsplit_0_mg_levels_ksp_richardson_scale": 0.8,
    "fieldsplit_0_mg_levels_pc_type": "jacobi",
    "fieldsplit_0_mg_coarse_ksp_type": "preonly",
    "fieldsplit_0_mg_coarse_pc_type": "python",
    "fieldsplit_0_mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_mg_coarse_assembled_pc_type": "redundant",
    "fieldsplit_0_mg_coarse_assembled_sub_pc_type": "lu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "firedrake.MassInvPC",
    "fieldsplit_1_Mp_ksp_type": "preonly",
    "fieldsplit_1_Mp_pc_type": "asm",
    "fieldsplit_1_Mp_sub_pc_type": "ilu"
}

up.assign(0)
solve(F == 0, up, bcs=bcs, solver_parameters=parameters, options_prefix='s')

if args.o:
    u,p = up.split()
    u.rename('velocity')
    p.rename('pressure')
    File(args.o).write(u,p)

