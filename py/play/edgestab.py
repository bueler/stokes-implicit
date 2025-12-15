#!/usr/bin/env python3

# Test linear edge stabilization from
#   E. Burman and P. Hansbo (2004). Edge stabilization for Galerkin approximations
#   of convection–diffusion–reaction problems. Computer Methods in Applied Mechanics
#   and Engineering 193, 1437--1453
# on a linear, steady reaction-diffusion-advection equation.  This works.
# However, in case 3 below, for an internal boundary layer we see a Gibbs effect,
# and for an outflow boundary layer we see an instability.  These are expected
# (Burmann & Hansbo 2004) because the method is applied without shock capturing.

# We also test weighted edge stabilization from
#   A. Ern and J. L. Guermond (2013). Weighting the edge stabilization.
#   SIAM Journal on Numerical Analysis 51(3), 1655--1677
# but it doesn't really make sense here for a steady equation, probably.  We are
# interpreting equations (3.3) and (3.4) in Ern & Guermond (2013) as a nonlinear
# PDE which we solve with a Picard iteration.  It does not work.

from firedrake import *
from firedrake.petsc import PETSc

printpar = PETSc.Sys.Print

stab = "es"  # default to linear edge stabilization
case = 3  # see below for cases
m = 40  # m x m mesh on unit square

stabtypes = ["none", "es", "wes"]

# mesh and function space
mesh = UnitSquareMesh(m - 1, m - 1, diagonal="crossed")
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", degree=1)
u = Function(V)  # initialized to zero here
v = TestFunction(V)

# set up particular problem
if case == 1:
    # Test Case 1 from Burman & Hansbo (2004)
    sig = 1.0
    bet = as_vector([1.0, 0.0])
    eps = 1.0e-5
    aw = 0.2
    uexact_ufl = exp(-((x - 0.5) ** 2 + 3.0 * (y - 0.5) ** 2) / aw)
    uexact = Function(V, name="uexact").interpolate(uexact_ufl)
    gbdry = uexact
    fsource_ufl = (
        sig * uexact_ufl - div(eps * grad(uexact_ufl)) + dot(bet, grad(uexact_ufl))
    )
    fsource = Function(V).interpolate(fsource_ufl)
elif case == 3:
    # Test Case from Burman & Hansbo (2004), shown in Figures 3--7
    sig = 0.0
    radtheta = (180.0 + 55.0) * pi / 180.0  # see Figure 3
    bet = as_vector([cos(radtheta), sin(radtheta)])
    eps = 1.0e-5
    gbdry = Function(V).interpolate(conditional(y - 0.8 * x > 0, 1.0, 0.0))
    fsource = Constant(0.0)
else:
    raise NotImplementedError

# weak form standard Galerkin approx
F = (sig * u * v + eps * dot(grad(u), grad(v)) + dot(bet, grad(u)) * v) * dx
F -= fsource * v * dx

# default solver params
params = {
    "snes_type": "ksponly",
    "ksp_type": "cg",
    "ksp_converged_reason": None,
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# set up edge stabilization
assert stab in stabtypes
if stab != "none":
    # linear edge stabilization is equation (3) in Burman & Hansbo (2004)
    n = FacetNormal(mesh)
    gamma = 0.025
    h0 = 1.0 / m
    hbdry = (2.0 + sqrt(2.0)) * h0  # total length of boundary of element
    CJ = 0.5 * gamma * hbdry**2
    J = CJ * jump(grad(u), n) * jump(grad(v), n) * dS

# Dirichlet boundary conditions from exact solution
bdry_ids = (1, 2, 3, 4)  # all four sides of boundary
bc = DirichletBC(V, gbdry, bdry_ids)

# solve
if stab == "none":
    solve(F == 0, u, bcs=[bc], options_prefix="s", solver_parameters=params)
elif stab == "es":
    F += J
    solve(F == 0, u, bcs=[bc], options_prefix="s", solver_parameters=params)
elif stab == "wes":
    # add weighted edge stabilization as an unsuccessful Picard iteration
    # note alpha(r) = 2 (1+r^2)^{-1} in equation (3.3); see page 1669.
    F += J
    DG0 = FunctionSpace(mesh, "DG", 0)
    for k in range(5):  # not clear how to iterate or how many times
        solve(F == 0, u, bcs=[bc], options_prefix="s", solver_parameters=params)
        alpha = Function(DG0).interpolate(2.0 / (1.0 + dot(grad(u), grad(u))))
        J = CJ * avg(alpha) * jump(grad(u), n) * jump(grad(v), n) * dS
        F = (sig * u * v + eps * dot(grad(u), grad(v)) + dot(bet, grad(u)) * v) * dx
        F -= fsource * v * dx
        F += J

oname = "result.pvd"
printpar("saving solution to %s ..." % oname)
u.rename("u")
if case == 1:
    printpar(f"m = {m}: |u-uexact|_L2 = {errornorm(uexact, u, 'L2'):.3e}")
    udiff = Function(V, name="u - uexact").interpolate(u - uexact)
    VTKFile(oname).write(u, uexact, udiff)
else:
    VTKFile(oname).write(u)
