# see https://www.firedrakeproject.org/demos/poisson.py.html
# (see also https://www.firedrakeproject.org/matrix-free.html)

# CHANGES: stdout & comments mods/declutter

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

## First, a direct solve with an assembled operator.::
#print('direct solve (LU) with assembled matrix ...')
#solve(a == L, uu, bcs=bcs, solver_parameters={"ksp_type": "preonly",
#                                              "pc_type": "lu"})

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
                                              "pc_type": "python",
                                              "pc_python_type": "firedrake.AssembledPC",
                                              "assembled_pc_type": "ilu"})

