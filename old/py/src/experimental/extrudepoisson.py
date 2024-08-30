#!/usr/bin/env python3

# for example,
#    ./extrudepoisson.py -s_snes_monitor -s_ksp_converged_reason -s_pc_type mg -semirefine -refine 2
# shows semicoarsening only in the base mesh direction

from firedrake import *
import sys

import argparse
parser = argparse.ArgumentParser(description='''
Extrude a 1D mesh to a unit square and solve Poisson on it.  The purpose is to
experiment with semicoarsening in extruded meshes.''',add_help=False)
parser.add_argument('-refine', type=int, default=1, metavar='N',
                    help='number of times to refine')
parser.add_argument('-semirefine', action='store_true', default=False,
                    help='FIXME')
parser.add_argument('-extrudepoissonhelp', action='store_true', default=False,
                    help='print help and quit')
args, unknown = parser.parse_known_args()
if args.extrudepoissonhelp:
    parser.print_help()
    sys.exit(0)

N = 2
base_coarse = UnitIntervalMesh(N)
base_hierarchy = MeshHierarchy(base_coarse, args.refine)
if args.semirefine:
    hierarchy = SemiCoarsenedExtrudedHierarchy(base_coarse, 1.0, base_layer=1, nref=args.refine)
else:
    hierarchy = ExtrudedMeshHierarchy(base_hierarchy, 1.0, base_layer=N)
#for k in range(args.refine+1):
#    print(hierarchy[k].coordinates.dat.data)
mesh = hierarchy[-1]
x, y = SpatialCoordinate(mesh)

H1 = FunctionSpace(mesh, 'CG', 1)
f = Function(H1)
f.interpolate(8.0 * pi * pi * cos(2 * pi * x) * cos(2 * pi * y))

g = Function(H1)
g.interpolate(cos(2 * pi * x) * cos(2 * pi * y))  # boundary condition and exact solution

u = Function(H1)
v = TestFunction(H1)
F = (dot(grad(v), grad(u)) - f * v) * dx

params = {'snes_type': 'ksponly',
          'ksp_type': 'cg'}
solve(F == 0, u, bcs=[DirichletBC(H1,g,1)], solver_parameters=params,
      options_prefix='s')

u.rename('u')
File('ep.pvd').write(u)

