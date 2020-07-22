# test EquationBC functionality in firedrake

# origin: https://github.com/firedrakeproject/firedrake/blob/master/tests/equation_bcs/test_equation_bcs.py

# see also Lawrence, Patrick comments on slack; alternative approaches:
#   Nitsche's method: https://scicomp.stackexchange.com/questions/19910/what-is-the-general-idea-of-nitsches-method-in-numerical-analysis
#   mixed form of Poisson: https://www.firedrakeproject.org/demos/poisson_mixed.py.html

from firedrake import *

porder = 1

solver_parameters = {'ksp_type': 'preonly',
                     'pc_type': 'lu'}

print('linear version:')
for mesh_num in [5,10,20,40]:
    mesh = UnitSquareMesh(mesh_num, mesh_num)
    V = FunctionSpace(mesh, "CG", porder)
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(- 8.0 * pi * pi * cos(x * pi * 2)*cos(y * pi * 2))

    a = dot(grad(v), grad(u)) * dx
    L = - f * v * dx

    g = Function(V)
    g.interpolate(cos(2 * pi * x) * cos(2 * pi * y))

    u_ = Function(V)

    bc1 = EquationBC(v * u * ds(1) == v * g * ds(1), u_, 1)

    solve(a == L, u_, bcs=[bc1], solver_parameters=solver_parameters)

    uexact = Function(V)
    uexact.interpolate(cos(x * pi * 2) * cos(y * pi * 2))
    #return sqrt(assemble(dot(u_ - f, u_ - f) * dx))
    print(sqrt(assemble(dot(u_ - uexact, u_ - uexact) * dx)))

solver_parameters = {'snes_type': 'newtonls',
                     'ksp_type': 'preonly',
                     'pc_type': 'lu'}

print('nonlinear version:')
for mesh_num in [5,10,20,40]:
    mesh = UnitSquareMesh(mesh_num, mesh_num)

    V = FunctionSpace(mesh, "CG", porder)

    u = Function(V)
    v = TestFunction(V)

    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(- 8.0 * pi * pi * cos(x * pi * 2) * cos(y * pi * 2))

    a = dot(grad(v), grad(u)) * dx
    L = - f * v * dx

    g = Function(V)
    g.interpolate(cos(2 * pi * x) * cos(2 * pi * y))

    if True:
        bc1 = EquationBC(v * (u - g) * ds(1) == 0, u, 1)
    else:
        # Equivalent to bc1 = EquationBC(v * (u - g1) * ds(1) == 0, u, 1)
        e2 = as_vector([0., 1.])
        bc1 = EquationBC((-dot(grad(v), e2) * dot(grad(u), e2) + 3 * pi * pi * v * u + 1 * pi * pi * v * g) * ds(1) == 0, u, 1)

    solve(a - L == 0, u, bcs=[bc1], solver_parameters=solver_parameters)

    uexact = Function(V)
    uexact.interpolate(cos(x * pi * 2) * cos(y * pi * 2))
    #return sqrt(assemble(dot(u - f, u - f) * dx))
    print(sqrt(assemble(dot(u - uexact, u - uexact) * dx)))

