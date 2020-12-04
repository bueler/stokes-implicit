#!/usr/bin/env python3

import sys, argparse
from firedrake import *
PETSc.Sys.popErrorHandler()

parser = argparse.ArgumentParser(description='''
Build-my-own mixed solution of Poisson based on reading Boffi et al 2013
pp. 22-25.
''',add_help=False)
parser.add_argument('-mphelp', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-N', type=int, default=8, metavar='N',
                    help='build NxN regular mesh of triangles (default=8)')
parser.add_argument('-o', metavar='FILE.pvd', type=str, default='',
                    help='save results to .pvd file')
args, unknown = parser.parse_known_args()
if args.mphelp:
    parser.print_help()
    sys.exit(0)

mesh = UnitSquareMesh(args.N, args.N)

V = VectorFunctionSpace(mesh, 'DP', 0)
Q = FunctionSpace(mesh, 'P', 1)
Z = V * Q

up = Function(Z)
u,p = split(up)
v,q = TestFunctions(Z)

x,y = SpatialCoordinate(mesh)
pexact = Function(Q).interpolate(sin(pi*x)*sin(2.0*pi*y))
f = Function(Q)
f = 5.0*pi*pi*pexact

F = (inner(u,v) - inner(v,grad(p)) - inner(u,grad(q)) + f*q) * dx
bcs = [DirichletBC(Z.sub(1), Constant(0.0), (1,2,3,4))]

params = {"ksp_type": "preonly",
          "pc_type": "lu",
          "ksp_converged_reason": None}

solve(F == 0, up, bcs=bcs, solver_parameters=params, options_prefix='s')

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

