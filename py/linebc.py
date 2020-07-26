#!/usr/bin/env python3

# origin: https://github.com/firedrakeproject/firedrake/blob/master/tests/equation_bcs/test_equation_bcs.py
# see also Lawrence, Patrick comments on slack; alternative approaches:
#   Nitsche's method: https://scicomp.stackexchange.com/questions/19910/what-is-the-general-idea-of-nitsches-method-in-numerical-analysis
#   mixed form of Poisson: https://www.firedrakeproject.org/demos/poisson_mixed.py.html

# Usage
# help:
#     ./linebc.py -h
# to generate small matrix:
#     ./linebc.py -m 2 -s_ksp_view_mat :foo.m:ascii_matlab
# convergence:
#     for m in 2 4 8 16 32; do ./linebc.py -m $m; done
# compare matrices from bc:
#     ./linebc.py -m 1 -s_ksp_view_mat ::ascii_dense
#     ./linebc.py -m 1 -s_ksp_view_mat ::ascii_dense -dirichletbc

from firedrake import *
import sys
from argparse import ArgumentParser, RawTextHelpFormatter


# needed for useful error messages
PETSc.Sys.popErrorHandler()

# process options
parser = ArgumentParser(\
    description='Test linear-solver EquationBC() functionality in firedrake.  Solver option prefix -s_',
    formatter_class=RawTextHelpFormatter)
parser.add_argument('-dirichletbc', action='store_true', default=False,
                    help='switch to using DirichletBC()')
parser.add_argument('-m', type=int, default=5, metavar='M',
                    help='number of elements in each direction')
parser.add_argument('-o', metavar='NAME', type=str, default='',
                    help='output file name ending with .pvd')
args, unknown = parser.parse_known_args()

porder = 1
solver_parameters = {'ksp_type': 'preonly',
                     'pc_type': 'lu'}

mesh = UnitSquareMesh(args.m, args.m)
V = FunctionSpace(mesh, "CG", porder)
u = TrialFunction(V)
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)
# right-hand side
f = Function(V)
f.interpolate(- 8.0 * pi * pi * cos(x * pi * 2)*cos(y * pi * 2))
# boundary condition and exact solution
g = Function(V)
g.interpolate(cos(2 * pi * x) * cos(2 * pi * y))

a = dot(grad(v), grad(u)) * dx
L = - f * v * dx

u_ = Function(V)
if args.dirichletbc:
    bc1 = DirichletBC(V,g,1)
else:
    bc1 = EquationBC(v * u * ds(1) == v * g * ds(1), u_, 1)
    # the following form is o.k. for nonlinear solver but not for linear:
    #    bc1 = EquationBC(v * (u-g) * ds(1) == 0, u_, 1)
solve(a == L, u_, bcs=[bc1], solver_parameters=solver_parameters,
      options_prefix='s')

print('l2 error for m=%d:  %.5f' % (args.m,sqrt(assemble(dot(u_ - g, u_ - g) * dx))))

if len(args.o) > 0:
    print('writing file %s ...' % args.o)
    u_.rename('numerical solution')
    g.rename('exact solution')
    outfile = File(args.o)
    outfile.write(u_,g)

