#!/usr/bin/env python3

# Usage help:
#     ./linebc.py -h
# To generate small matrix:
#     ./linebc.py -m 2 -s_ksp_view_mat :foo.m:ascii_matlab
# Convergence using AMG:
#     for m in 20 40 80 160; do ./linebc.py -m $m -s_pc_type gamg; done
# Compare matrices from bc:
#     ./linebc.py -m 1 -s_ksp_view_mat ::ascii_dense
#     ./linebc.py -m 1 -s_ksp_view_mat ::ascii_dense -dirichletbc

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
    description='Test linear-solver EquationBC() functionality in firedrake.  Solver option prefix -s_')
parser.add_argument('-dirichletbc', action='store_true', default=False,
                    help='switch to using DirichletBC()')
parser.add_argument('-m', type=int, default=5, metavar='M',
                    help='number of elements in each direction')
parser.add_argument('-k', type=int, default=1, metavar='K',
                    help='polynomial degree of Pk elements')
parser.add_argument('-o', metavar='NAME', type=str, default='',
                    help='output file name ending with .pvd')
args, unknown = parser.parse_known_args()

solver_parameters = {'ksp_type': 'gmres',
                     'pc_type': 'lu'}

mesh = UnitSquareMesh(args.m, args.m)
V = FunctionSpace(mesh, "CG", args.k)
u = TrialFunction(V)
v = TestFunction(V)

# right-hand side
x, y = SpatialCoordinate(mesh)
f = Function(V)
f.interpolate(- 8.0 * pi * pi * cos(x * pi * 2)*cos(y * pi * 2))

# boundary condition and exact solution
g = Function(V)
g.interpolate(cos(2 * pi * x) * cos(2 * pi * y))

# set up weak form
a = dot(grad(v), grad(u)) * dx
L = - f * v * dx

# apply boundary condition only on edge 1
u_ = Function(V)   # initialized to zero
if args.dirichletbc:
    bc1 = DirichletBC(V,g,1)
else:
    bc1 = EquationBC(v * u * ds(1) == v * g * ds(1), u_, 1)
    # the following form is o.k. for nonlinear solver but not for linear:
    #    bc1 = EquationBC(v * (u-g) * ds(1) == 0, u_, 1)

# use linear solver
solve(a == L, u_, bcs=[bc1], solver_parameters=solver_parameters,
      options_prefix='s')

print('l2 error for m=%d:  %.3e' % (args.m,sqrt(assemble(dot(u_ - g, u_ - g) * dx))))

if len(args.o) > 0:
    print('writing file %s ...' % args.o)
    u_.rename('numerical solution')
    g.rename('exact solution')
    outfile = File(args.o)
    outfile.write(u_,g)

