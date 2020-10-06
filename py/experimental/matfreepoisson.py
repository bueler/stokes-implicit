# see https://www.firedrakeproject.org/demos/poisson.py.html
# (see also https://www.firedrakeproject.org/matrix-free.html)

# CHANGES: stdout mods

# Poisson equation
# ================
#
# It is what it is, a conforming discretization on a regular mesh using
# piecewise quadratic elements.
#
# As usual we start by importing firedrake and setting up the problem.::

from firedrake import *

N = 128

mesh = UnitSquareMesh(N, N)

V = FunctionSpace(mesh, "CG", 2)

u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u), grad(v)) * dx

x = SpatialCoordinate(mesh)
F = Function(V)
F.interpolate(sin(x[0]*pi)*sin(2*x[1]*pi))
L = F*v*dx

bcs = [DirichletBC(V, Constant(2.0), (1,))]

uu = Function(V)

# With the setup out of the way, we now demonstrate various ways of
# configuring the solver.  First, a direct solve with an assembled
# operator.::

print('direct solve (LU) with assembled matrix ...')
solve(a == L, uu, bcs=bcs, solver_parameters={"ksp_type": "preonly",
                                              "pc_type": "lu"})

# Next, we use unpreconditioned conjugate gradients using matrix-free
# actions.  This is not very efficient due to the :math:`h^{-2}`
# conditioning of the Laplacian, but demonstrates how to request an
# unassembled operator using the ``"mat_type"`` solver parameter.::

print('matrix-free with unpreconditioned CG ...')
uu.assign(0)
solve(a == L, uu, bcs=bcs, solver_parameters={"mat_type": "matfree",
                                              "ksp_type": "cg",
                                              "pc_type": "none",
                                              "ksp_converged_reason": None})

# Finally, we demonstrate the use of a :class:`.AssembledPC`
# preconditioner.  This uses matrix-free actions but preconditions the
# Krylov iterations with an incomplete LU factorisation of the assembled
# operator.::

print('matrix-free with CG and assembled ILU PC ...')
uu.assign(0)
solve(a == L, uu, bcs=bcs, solver_parameters={"mat_type": "matfree",
                                              "ksp_type": "cg",
                                              "ksp_converged_reason": None,

# To use the assembled matrix for the preconditioner we select a
# ``"python"`` type::

                                              "pc_type": "python",

# and set its type, by providing the name of the class constructor to
# PETSc.::

                                              "pc_python_type": "firedrake.AssembledPC",

# Finally, we set the preconditioner type for the assembled operator::

                                              "assembled_pc_type": "ilu"})

# This demo is available as a runnable python file `here
# <poisson.py>`__.
