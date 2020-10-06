#!/usr/bin/env python3

# for example,
#    ./extrudepoisson.py -s_snes_monitor -s_ksp_converged_reason -s_pc_type mg -semirefine -refine 2
# shows semicoarsening only in the base mesh direction

from firedrake import *
import sys

import argparse
parser = argparse.ArgumentParser(description='''
Extrude a 1D mesh to a unit square and solve Poisson on it.  The purpose is to
experiment with semicoarsening in extruded meshes.''',
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 add_help=False)
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
    hierarchy = ExtrudedMeshHierarchy(base_hierarchy, 1.0, base_layer=(2**args.refine)*N, refinement_ratio=1)
else:
    hierarchy = ExtrudedMeshHierarchy(base_hierarchy, 1.0, base_layer=N)
for k in range(2):
    print(hierarchy[k].coordinates.dat.data)
mesh = hierarchy[-1]

P1 = FiniteElement("CG", interval, 1)
H1_element = TensorProductElement(P1, P1)
H1 = FunctionSpace(mesh, H1_element)

x, y = SpatialCoordinate(mesh)
f = Function(H1)
f.interpolate(8.0 * pi * pi * cos(x * pi * 2)*cos(y * pi * 2))

# boundary condition and exact solution
g = Function(H1)
g.interpolate(cos(2 * pi * x) * cos(2 * pi * y))
bc1 = DirichletBC(H1,g,1)

u = Function(H1)   # initialized to zero
v = TestFunction(H1)
F = (dot(grad(v), grad(u)) - f * v) * dx

params = {'snes_type': 'ksponly',
          'ksp_type': 'cg'}
solve(F == 0, u, bcs=[bc1], solver_parameters=params,
      options_prefix='s')

filename = 'ep.pvd'
print('writing file %s ...' % filename)
u.rename('u')
File(filename).write(u)

