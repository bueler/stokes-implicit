# boring Poisson example to test SemiCoarsenExtrudedHierarchy() with GMG
# generated mesh is 10 x 4 quad elements with 10 x 1 coarse gmesh
# example to show multigrid v-cycles:
#   python3 semicoarsenpoisson.py -s_ksp_converged_reason -s_mg_levels_ksp_converged_reason -s_mg_coarse_ksp_converged_reason

from firedrake import *

N = 10
base = UnitIntervalMesh(N)
hierarchy = SemiCoarsenedExtrudedHierarchy(base, 1.0, base_layer=1, nref=2)
mesh = hierarchy[-1]

x, y = SpatialCoordinate(mesh)

H1 = FunctionSpace(mesh, 'CG', 1)
f = Function(H1).interpolate(8.0 * pi * pi * cos(2 * pi * x) * cos(2 * pi * y))
g = Function(H1).interpolate(cos(2 * pi * x) * cos(2 * pi * y))  # boundary condition and exact solution

u = Function(H1)
v = TestFunction(H1)
F = (dot(grad(v), grad(u)) - f * v) * dx

params = {'snes_type': 'ksponly',
          'ksp_type': 'cg',
          'pc_type': 'mg'}
solve(F == 0, u, bcs=[DirichletBC(H1,g,1)], solver_parameters=params,
      options_prefix='s')

u.rename('u')
File('scp.pvd').write(u)

