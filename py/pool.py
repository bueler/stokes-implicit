#!/usr/bin/env python3

# preliminary success 1: constant iterations and sublinear time:
#   for X in 2 3 4 5 6; do tmpg -n 6 ./pool.py -s_ksp_rtol 1.0e-4 -mx 20 -my 20 -refine $X -s_ksp_converged_reason; done

# preliminary success 2: over 1e7 degrees of freedom (80 x 80 x 64 elements) with element aspect ratio 8, in under 600 seconds:
#   tmpg -n 8 ./pool.py -s_ksp_rtol 1.0e-4 -mx 80 -my 80 -refine 6 -s_ksp_converged_reason

from firedrake import *
import sys, argparse

parser = argparse.ArgumentParser(description='''
Lid-driven Stokes in 3D domain with high aspect ratio, i.e. large horizontal
and small vertical dimensions.  Applies SemiCoarsenExtrudedHierarchy() to
to refine the vertical direction.  Uses Q2xQ1 mixed elements.  Solver is
Schur fieldsplit with vertical-coarsening-only GMG on the velocity block.''',
           add_help=False)
parser.add_argument('-direct', action='store_true', default=False,
                    help='use a monolithic direct solver')
parser.add_argument('-mx', type=int, default=10, metavar='N',
                    help='number of equal subintervals in x-direction (default=10)')
parser.add_argument('-my', type=int, default=10, metavar='N',
                    help='number of equal subintervals in y-direction (default=10)')
parser.add_argument('-mz', type=int, default=1, metavar='N',
                    help='number of layers in each vertical column (default=1)')
parser.add_argument('-o', metavar='FILE.pvd', type=str, default='',
                    help='save to output file name ending with .pvd')
parser.add_argument('-poolhelp', action='store_true', default=False,
                    help='print help for stokes2D.py and quit')
parser.add_argument('-refine', type=int, default=2, metavar='N',
                    help='number of vertical (z) mesh refinements (default=2)')
parser.add_argument('-schurcoarsegamg', action='store_true', default=False,
                    help='use AMG solver gamg on coarse grid problem for velocity block')
parser.add_argument('-schurdirect', action='store_true', default=False,
                    help='use a direct solver composed with Schur fieldsplit')
args, unknown = parser.parse_known_args()
if args.poolhelp:
    parser.print_help()
    sys.exit(0)

L, H = 10.0, 1.0  # domain fixed at 10x10x1 lengths
base = RectangleMesh(args.mx,args.my,L,L,quadrilateral=True)
hierarchy = SemiCoarsenedExtrudedHierarchy(base,H,base_layer=args.mz,nref=args.refine)
mesh = hierarchy[-1]
fmz = args.mz * 2**args.refine
PETSc.Sys.Print('mesh:                %d x %d x %d elements (hexs)' \
                % (args.mx,args.my,fmz))
dxelem, dyelem, dzelem = L / args.mx, L / args.my, H / fmz
PETSc.Sys.Print('element dims:        %.3f x %.3f x %.3f with ratiox=%.2f, ratioy=%.2f' \
                % (dxelem, dyelem, dzelem, dxelem/dzelem, dyelem/dzelem))

V = VectorFunctionSpace(mesh, 'CG', 2)
W = FunctionSpace(mesh, 'CG', 1)
Z = V * W

n_u,n_p,N = V.dim(),W.dim(),Z.dim()
PETSc.Sys.Print('vector space dims:   n_u=%d, n_p=%d  -->  N=%d' % (n_u,n_p,N))

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

F = (inner(grad(u), grad(v)) - p * div(v) + div(u) * q - inner(Constant((0, 0, 0)), v))*dx

# velocity on lid is (1,1), i.e. in both x and y directions
bcs = [DirichletBC(Z.sub(0), Constant((1.0, 1.0, 0)), 'top'),
       DirichletBC(Z.sub(0), Constant((0, 0, 0)), (1, 2, 3, 4)),
       DirichletBC(Z.sub(0), Constant((0, 0, 0)), 'bottom')]

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

params = {'snes_type': 'ksponly'}
if args.direct:
    params['mat_type'] = 'aij'
    params['ksp_type'] = 'preonly'
    params['pc_type'] = 'lu'
    params['pc_factor_mat_solver_type'] = 'mumps'
else:
    if args.schurcoarsegamg:
        params['mat_type'] = 'aij'
    else:
        params['mat_type'] = 'nest'
    params['ksp_type'] = 'gmres'
    params['pc_type'] = 'fieldsplit'
    params['pc_fieldsplit_type'] = 'schur'
    params['pc_fieldsplit_schur_fact_type'] = 'lower'
    params['pc_fieldsplit_schur_precondition'] = 'selfp'
    params['fieldsplit_0_ksp_type'] = 'preonly'
    params['fieldsplit_1_ksp_type'] = 'preonly'
    #params['fieldsplit_1_ksp_type'] = 'richardson'
    #params['fieldsplit_1_ksp_max_it'] = 3
    #params['fieldsplit_1_ksp_converged_maxits'] = None
    #params['fieldsplit_1_ksp_richardson_scale'] = 1.0
    params['fieldsplit_1_pc_type'] = 'jacobi'
    params['fieldsplit_1_pc_jacobi_type'] = 'diagonal'
    #params['fieldsplit_1_ksp_type'] = 'gmres'
    #params['fieldsplit_1_ksp_rtol'] = 0.1
    #params['fieldsplit_1_pc_type'] = 'bjacobi'
    #params['fieldsplit_1_sub_pc_type'] = 'ilu'
    if args.schurdirect:
        params['fieldsplit_0_pc_type'] = 'lu'
    else:
        params['fieldsplit_0_pc_type'] = 'mg'
        params['fieldsplit_0_mg_coarse_ksp_type'] = 'preonly'
        if args.schurcoarsegamg:
            params['fieldsplit_0_mg_coarse_pc_type'] = 'gamg'
        else:
            params['fieldsplit_0_mg_coarse_pc_type'] = 'lu'

#print(params)

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

