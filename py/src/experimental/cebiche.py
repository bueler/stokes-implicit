#!/usr/bin/env python3

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
where u is a vector, p is scalar, and g(x,y) is a given function.  We have
boundary condition
       u . n = 0
Note that u is a vector field and that the scalar Poisson equation is
div(grad p) = g  with Neumann boundary condition  dp/dn = 0.  The default
primal weak form is
  <u,v> - <grad p,v> = 0  for all v in (L^2)^2
        - <u,grad q> = 0  for all q in H^1 s.t. int_Omega q = 0
The dual (-dual) weak form is
   <u,v> + <p,div v> = 0  for all v in H^1_0(div)
           <div u,q> = 0  for all q in L^2 s.t. int_Omega q = 0
An exact solution based on
  p(x,y) = cos(pi x) cos(2 pi y)
is used, with differentiation giving u and then g.  Note g satisfies
int_Omega g = 0, as implied by the Neumann condition.
''',add_help=False)
parser.add_argument('-cebichehelp', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-dual', action='store_true', default=False,
                    help='use dual weak form over H(div) x L^2')
parser.add_argument('-N', type=int, default=4, metavar='N',
                    help='build NxN regular mesh of triangles (default=8)')
parser.add_argument('-o', metavar='FILE.pvd', type=str, default='',
                    help='save results to .pvd file')
args, unknown = parser.parse_known_args()
if args.cebichehelp:
    parser.print_help()
    sys.exit(0)

mesh = UnitSquareMesh(args.N, args.N) # mesh of triangles
x,y = SpatialCoordinate(mesh)

if args.dual:
    E = FiniteElement('BDM',triangle,1,variant='integral')  # suppress warning
    V = FunctionSpace(mesh, E)
    Q = FunctionSpace(mesh, 'DP', 0)
else:
    V = VectorFunctionSpace(mesh, 'DP', 0)
    Q = FunctionSpace(mesh, 'P', 1)
Z = V * Q

# FIXME also try "project"?
pexact = Function(Q).interpolate(cos(pi*x)*cos(2.0*pi*y))
uexact = Function(V).interpolate(as_vector([-pi*sin(pi*x)*cos(2.0*pi*y),
                                            -2.0*pi*cos(pi*x)*sin(2.0*pi*y)]))
g = Function(Q).interpolate(-5.0*pi*pi*pexact)

up = Function(Z)
u,p = split(up)
v,q = TestFunctions(Z)

params = {"snes_type": "ksponly",
          "ksp_type": "preonly",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps",
          "ksp_converged_reason": None}

if args.dual:
    F = (inner(u,v) + p * div(v) + div(u) * q - g*q) * dx
    bcs = [DirichletBC(Z.sub(0), Constant((0.0,0.0)), (1,2,3,4)),]
    solve(F == 0, up, bcs=bcs, solver_parameters=params, options_prefix='s')
else:
    F = (inner(u,v) - inner(grad(p),v) - inner(u,grad(q)) - g*q) * dx
    solve(F == 0, up, bcs=None, solver_parameters=params, options_prefix='s')

u,p = up.split()
udiff = Function(V).interpolate(u - uexact)
uerr_L2 = sqrt(assemble(inner(udiff, udiff) * dx))
pdiff = Function(Q).interpolate(p - pexact)
perr_L2 = sqrt(assemble(pdiff * pdiff * dx))
PETSc.Sys.Print('on %d x %d mesh:  |u-uexact|_2 = %.3e, |p-pexact|_2 = %.3e' \
                % (args.N,args.N,uerr_L2,perr_L2))

if args.o:
    u,p = up.split()
    u.rename('grad p')
    p.rename('p')
    File(args.o).write(u,p)

