#!/usr/bin/env python3

# test edge stabilization from
#   E. Burman and P. Hansbo, Edge stabilization for Galerkin approximations of
#   convection–diffusion–reaction problems, Computer Methods in Applied Mechanics
#   and Engineering, 193 (2004), 1437--1453

from firedrake import *

es = True  # edge stabilization
case = 3
m = 40

# mesh and function space
mesh = UnitSquareMesh(m-1, m-1, diagonal='crossed')
x,y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, 'Lagrange', degree=1)
u = Function(V)  # initialized to zero here
v = TestFunction(V)

# set up particular problem
if case == 1:
    # Test Case 1 from Burman & Hansbo (2004)
    sig = 1.0
    bet = as_vector([1.0, 0.0])
    eps = 1.0e-5
    aw = 0.2
    uexact_ufl = exp(- ((x - 0.5)**2 + 3.0 * (y-0.5)**2) / aw)
    uexact = Function(V, name='uexact').interpolate(uexact_ufl)
    gbdry = uexact
    fsource_ufl = sig * uexact_ufl - div(eps * grad(uexact_ufl)) + dot(bet, grad(uexact_ufl))
    fsource = Function(V).interpolate(fsource_ufl)
elif case == 3:
    # Test Case from Burman & Hansbo (2004) in Figures 3--7
    sig = 0.0
    radtheta = (180.0 + 55.0) * pi / 180.0  # see Figure 3
    bet = as_vector([cos(radtheta), sin(radtheta)])
    eps = 1.0e-5
    gbdry = Function(V).interpolate(conditional(y - 0.8 * x > 0, 1.0, 0.0))
    fsource = Constant(0.0)
else:
    raise NotImplementedError

# weak form standard Galerkin approx for reaction-diffusion-advection equation
F = (sig * u * v + eps * dot(grad(u), grad(v)) + dot(bet, grad(u)) * v) * dx \
    - fsource * v * dx

if es:
    # add edge stabilization
    gamma = 0.025
    hedge = 1.0 / m
    n = FacetNormal(mesh)
    J = 0.5 * gamma * hedge**2 * jump(grad(u),n) * jump(grad(v),n) * dS
    F += J

# Dirichlet boundary conditions from exact solution
bdry_ids = (1, 2, 3, 4)   # all four sides of boundary
bc = DirichletBC(V, gbdry, bdry_ids)

params = {'snes_type': 'ksponly',
          'ksp_type': 'cg',
          'ksp_converged_reason': None,
          }
solve(F == 0, u, bcs = [bc], options_prefix = 's',
      solver_parameters = params)

oname = "result.pvd"
print('saving solution to %s ...' % oname)
u.rename('u')
if case == 1:
    print(f'm = {m}: |u-uexact|_L2 = {errornorm(uexact, u, 'L2'):.3e}')
    udiff = Function(V, name='u - uexact').interpolate(u - uexact)
    VTKFile(oname).write(u, uexact, udiff)
else:
    VTKFile(oname).write(u)
