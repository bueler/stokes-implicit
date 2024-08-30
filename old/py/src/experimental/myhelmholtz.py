# simple Helmholtz equation

from firedrake import *
mesh = UnitSquareMesh(10, 10)

V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
a = (dot(grad(v), grad(u)) + v * u) * dx
L = f * v * dx

u = Function(V)
solve(a == L, u,
      solver_parameters={'ksp_converged_reason': None,
                         'ksp_type': 'cg',
                         'pc_type': 'icc'},
      options_prefix='s')

