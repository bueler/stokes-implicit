#!/usr/bin/env python3

# the primal and dual problems converges with the default MINRES solver:
#   $ for LEV in 2 4 8 16 32 64 128; do ./ceviche.py -N $LEV; done
#   $ for LEV in 2 4 8 16 32 64 128; do ./ceviche.py -dual -N $LEV; done
# the iterations are smaller for the primal problem but the errors are
# smaller for the dual

import sys, argparse
from firedrake import *
PETSc.Sys.popErrorHandler()

parser = argparse.ArgumentParser(description='''
Solve two mixed formulations of the Poisson.  (Roughly based on Examples 3.2
and 3.3 in Mardal & Winther 2011, also Examples 1.3.4, 1.3.5 on pages 22--25
of Boffi et al 2013.  Ceviche is a bowl of mixed seafood.)  The domain is
Omega = [0,1]^2.  The strong form is
          u - grad p = 0
               div u = g
with Neumann boundary condition  u . n = 0.  Here u is a vector, p is scalar,
and g(x,y) is a given function.  This problem implies the Poisson equation
div(grad p) = g  with  dp/dn = 0,  which is well-posed over p with mean zero.
An exact solution based on  p(x,y) = cos(pi x) cos(2 pi y)  is used,
differentiation giving u and then g (which has mean zero as needed.)  The
default primal weak form is
  <u,v> - <grad p,v> = 0  for all v in (L^2)^2
        - <u,grad q> = 0  for all q in H^1 s.t. int_Omega q = 0
The dual (-dual) weak form is
   <u,v> + <p,div v> = 0  for all v in H^1_0(div)
           <div u,q> = 0  for all q in L^2 s.t. int_Omega q = 0
The primal elements are DP0 x P1 and the dual elements are BDM1 x DP0.
The default solver is unpreconditioned MINRES.
''',
    add_help=False,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-cevichehelp', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-dual', action='store_true', default=False,
                    help='use dual weak form over H(div) x L^2')
parser.add_argument('-k', type=int, default=1, metavar='N',
                    help='element order, either DPk-1 x Pk or BDMk x DPk-1 (default=1)')
parser.add_argument('-N', type=int, default=4, metavar='N',
                    help='build NxN regular mesh of triangles (default=8)')
parser.add_argument('-o', metavar='FILE.pvd', type=str, default='',
                    help='save results to .pvd file')
parser.add_argument('-rt', action='store_true', default=False,
                    help='with -dual, use Raviart-Thomas elements instead of BDM')
args, unknown = parser.parse_known_args()
if args.cevichehelp:
    parser.print_help()
    sys.exit(0)

mesh = UnitSquareMesh(args.N, args.N) # mesh of triangles
x,y = SpatialCoordinate(mesh)

if args.dual:
    # note the integral variant is used to suppress "DeprecationWarning:
    #   Variant of X element will change from point evaluation to integral
    #   evaluation."
    E = FiniteElement('RT' if args.rt else 'BDM',triangle,args.k,variant='integral')
    V = FunctionSpace(mesh, E)
    Q = FunctionSpace(mesh, 'DP', args.k-1)
else:
    V = VectorFunctionSpace(mesh, 'DP', args.k-1)
    Q = FunctionSpace(mesh, 'P', args.k)
Z = V * Q

pexact = Function(Q).project(cos(pi*x)*cos(2.0*pi*y))
uexact = Function(V).project(as_vector([-pi*sin(pi*x)*cos(2.0*pi*y),
                                        -2.0*pi*cos(pi*x)*sin(2.0*pi*y)]))
# alternative for primal:  uexact = grad(pexact)
g = Function(Q).project(-5.0*pi*pi*pexact)

up = Function(Z)
u,p = split(up)
v,q = TestFunctions(Z)

params = {"snes_type": "ksponly",
          "ksp_type": "minres",
          "pc_type": "none",
          "ksp_converged_reason": None}

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
if args.dual:
    F = (inner(u,v) + p * div(v) + div(u) * q - g*q) * dx
    bcs = [DirichletBC(Z.sub(0), Constant((0.0,0.0)), (1,2,3,4)),]
    solve(F == 0, up, bcs=bcs, nullspace=nullspace,
          solver_parameters=params, options_prefix='s')
else:
    F = (inner(u,v) - inner(grad(p),v) - inner(u,grad(q)) - g*q) * dx
    solve(F == 0, up, bcs=None, nullspace=nullspace,
          solver_parameters=params, options_prefix='s')

u,p = up.split()
udiff = Function(V).interpolate(u - uexact)
uerr_L2 = sqrt(assemble(inner(udiff, udiff) * dx))
pdiff = Function(Q).interpolate(p - pexact)
perr_L2 = sqrt(assemble(pdiff * pdiff * dx))
PETSc.Sys.Print('on %3d x %3d mesh:  |u-uexact|_2 = %.3e, |p-pexact|_2 = %.3e' \
                % (args.N,args.N,uerr_L2,perr_L2))

if args.o:
    u,p = up.split()
    u.rename('u = grad p')
    p.rename('p')
    File(args.o).write(u,p)

