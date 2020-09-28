#!/usr/bin/env python3

# Usage help:
#     ./ebc.py -h
# The six solver/bc versions:
#     ./ebc.py                          # linear solver, weak dirichlet bc
#     ./ebc.py -dirichletbc             # linear solver, strong dirichlet bc
#     ./ebc.py -tangential              # linear solver, tangential ODE bc
#     ./ebc.py -nonlinear               # nonlinear solver, weak dirichlet bc
#     ./ebc.py -nonlinear -dirichletbc  # nonlinear solver, strong dirichlet bc
#     ./ebc.py -nonlinear -tangential   # nonlinear solver, tangential ODE bc
# (Note -dirichletbc -tangential is not allowed.)
# Convergence using AMG:
#     for m in 20 40 80 160; do ./ebc.py -m $m -s_pc_type gamg; done
# Save small matrix:
#     ./ebc.py -m 2 -s_ksp_view_mat :foo.m:ascii_matlab
# Compare matrices from bc versions:
#     ./ebc.py -m 1 -s_ksp_view_mat ::ascii_dense
#     ./ebc.py -m 1 -s_ksp_view_mat ::ascii_dense -dirichletbc
#     ./ebc.py -m 1 -s_ksp_view_mat ::ascii_dense -tangential

# method:  EquationBC() treats the boundary face (edge) as an L^2 space
#     and sets-up a weak form for the given boundary condition equation.
#     For example, for a Dirichlet condition u=g on boundary edge 1 of a
#     unit square, namely {x=0,0<y<1}, it is the weak form
#       \int_0^1 u(0,y) v(y) dy = \int_0^1 g(y) v(y) dy
#     With P1 elements and UnitSquareMesh(1,1), thus a triangulation of
#     the unit square with only two triangles, the representation of u(0,y)
#     is u(0,y) = c0 psi0(y) + c3 psi3(y).  Assuming
#     g(x,y)=cos(2 pi x)cos(2 pi y) as below, the equations become
#       (1/3) c0 + (1/6) c3 = 1/2
#       (1/6) c0 + (1/3) c3 = 1/2
#     (This ignores the fact that the system is actually A c = r, i.e.
#     residual form, and not A u = b.)

# origin: https://github.com/firedrakeproject/firedrake/blob/master/tests/equation_bcs/test_equation_bcs.py
# see also Lawrence, Patrick comments on slack; alternative approaches:
#   Nitsche's method: https://scicomp.stackexchange.com/questions/19910/what-is-the-general-idea-of-nitsches-method-in-numerical-analysis
#   mixed form of Poisson: https://www.firedrakeproject.org/demos/poisson_mixed.py.html

from firedrake import *
import sys
from argparse import ArgumentParser

# needed for useful error messages
PETSc.Sys.popErrorHandler()

# process options
parser = ArgumentParser(\
    description="Test Firedrake's EquationBC() functionality on a Poisson problem on the unit square.  By default uses linear form a==L.  Solver option prefix -s_.")
parser.add_argument('-dirichletbc', action='store_true', default=False,
                    help='switch to DirichletBC() instead of EquationBC()')
parser.add_argument('-k', type=int, default=1, metavar='K',
                    help='polynomial degree of Pk elements')
parser.add_argument('-m', type=int, default=5, metavar='M',
                    help='number of elements in each direction')
parser.add_argument('-nonlinear', action='store_true', default=False,
                    help='switch to nonlinear solver for F==0 instead of a==L')
parser.add_argument('-o', metavar='NAME', type=str, default='',
                    help='output file name ending with .pvd')
parser.add_argument('-tangential', action='store_true', default=False,
                    help='apply the tangential derivative boundary condition, an ODE u_yy + C1 u = C2 g')
args, unknown = parser.parse_known_args()

assert ((not args.dirichletbc) or (not args.tangential)), 'option combination not allowed'

mesh = UnitSquareMesh(args.m, args.m)
V = FunctionSpace(mesh, "CG", args.k)
v = TestFunction(V)

# right-hand side
x, y = SpatialCoordinate(mesh)
fRHS = Function(V)
fRHS.interpolate(- 8.0 * pi * pi * cos(x * pi * 2)*cos(y * pi * 2))

# boundary condition and exact solution
g = Function(V)
g.interpolate(cos(2 * pi * x) * cos(2 * pi * y))

# set up weak form a == L or F == 0
if args.nonlinear:
    u = Function(V)   # initialized to zero
    F = (dot(grad(v), grad(u)) + fRHS * v) * dx
else:
    u = TrialFunction(V)
    u_ = Function(V)   # initialized to zero
    a = dot(grad(v), grad(u)) * dx
    L = - fRHS * v * dx

# apply boundary condition only on edge 1
if args.tangential:
    # the ODE along the left/1 bdry:  u_yy + 3 pi^2 u = -pi^2 g
    e2 = as_vector([0., 1.])
    tl = - dot(grad(v), e2) * dot(grad(u), e2) + 3 * pi * pi * v * u
    tr = - pi * pi * v * g
if args.dirichletbc:
    bc1 = DirichletBC(V,g,1)
else:
    if args.nonlinear:  # nonlinear form of bcs
        if args.tangential:
            bc1 = EquationBC((tl - tr) * ds(1) == 0, u, 1)
        else:
            bc1 = EquationBC(v * (u - g) * ds(1) == 0, u, 1)
    else:               # linear form of bcs
        if args.tangential:
            bc1 = EquationBC(tl * ds(1) == tr * ds(1), u_, 1)
        else:
            bc1 = EquationBC(v * u * ds(1) == v * g * ds(1), u_, 1)

# default to a serial-only direct solver; usual PETSc solvers apply
solver_parameters = {'ksp_type': 'gmres',
                     'pc_type': 'lu'}
if args.nonlinear:
    solver_parameters.update({'snes_type': 'newtonls'})
    solve(F == 0, u, bcs=[bc1], solver_parameters=solver_parameters,
          options_prefix='s')
else:
    solve(a == L, u_, bcs=[bc1], solver_parameters=solver_parameters,
          options_prefix='s')
    u = u_

print('l2 error for m=%d:  %.3e' % (args.m,sqrt(assemble(dot(u - g, u - g) * dx))))

if len(args.o) > 0:
    print('writing file %s ...' % args.o)
    u.rename('numerical solution')
    g.rename('exact solution')
    outfile = File(args.o)
    outfile.write(u,g)

