#!/usr/bin/env python3

import sys, argparse
from firedrake import *
PETSc.Sys.popErrorHandler()

parser = argparse.ArgumentParser(description='''
Solve two mixed formulations of the Poisson, based on Examples 3.2 and 3.3
in Mardal & Winther 2011.  The domain is Omega = [0,1]^2.  The strong form is
  u - grad p = 0
       div u = g
where g(x,y) is a given function, and with boundary conditions
       u . n = 0
Note that u is a vector field and that the scalar Poisson equation is
div(grad p) = g  with Neumann boundary condition  dp/dn = 0.  The first
(default) weak form is
   <u,v> + <p,div v> = 0  for all v in H^1_0(div)
           <div u,q> = 0  for all q in L^2 s.t. int_Omega q = 0
The second (-alt) weak form is
  <u,v> - <grad p,v> = 0  for all v in (L^2)^2
        - <u,grad q> = 0  for all q in H^1 s.t. int_Omega q = 0
An exact solution based on
  p(x,y) = cos(pi x) cos(2 pi y)
is used, with differentiation giving u and then g.
''',add_help=False)
parser.add_argument('-alt', action='store_true', default=False,
                    help='use alternative weak form')
parser.add_argument('-mphelp', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-N', type=int, default=4, metavar='N',
                    help='build NxN regular mesh of triangles (default=8)')
parser.add_argument('-o', metavar='FILE.pvd', type=str, default='',
                    help='save results to .pvd file')
args, unknown = parser.parse_known_args()
if args.mphelp:
    parser.print_help()
    sys.exit(0)

mesh = UnitSquareMesh(args.N, args.N)
x,y = SpatialCoordinate(mesh)

if args.alt:
    V = VectorFunctionSpace(mesh, 'DP', 0)
    Q = FunctionSpace(mesh, 'P', 1)
else:
    E = FiniteElement('BDM',triangle,1,variant='integral')  # suppress warning
    V = FunctionSpace(mesh, E)
    Q = FunctionSpace(mesh, 'DG', 0)
Z = V * Q

# FIXME also try "project"
pexact = Function(Q).interpolate(cos(pi*x)*cos(2.0*pi*y))
g = Function(Q).interpolate(-5.0*pi*pi*pexact)

up = Function(Z)
u,p = split(up)
v,q = TestFunctions(Z)

params = {"snes_type": "ksponly",
          "ksp_type": "preonly",
          #"pc_type": "svd",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps",
          "ksp_converged_reason": None}

if args.alt:
    F = (inner(u,v) - inner(grad(p),v) - inner(u,grad(q)) - g*q) * dx
    solve(F == 0, up, bcs=None, solver_parameters=params, options_prefix='s')
else:
    F = (dot(u,v) + p * div(v) + div(u) * q - g*q) * dx
    bcs = [DirichletBC(Z.sub(0), Constant((0.0,0.0)), (1,2,3,4)),]
    #solve(F == 0, up, bcs=bcs, solver_parameters=params, options_prefix='s')
    solve(F == 0, up, bcs=bcs, options_prefix='s')

u,p = up.split()
pdiff = Function(Q).interpolate(p - pexact)
error_L2 = sqrt(assemble(dot(pdiff, pdiff) * dx))
PETSc.Sys.Print('on %d x %d mesh:  |p-pexact|_h = %.3e' \
                % (args.N,args.N,error_L2))

if args.o:
    u,p = up.split()
    u.rename('grad p')
    p.rename('p')
    File(args.o).write(u,p)

