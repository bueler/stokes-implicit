# Simplified Navier-Stokes solver for lid-driven cavity using P2-P1
# Taylor-Hood elements and a direct solver (MUMPS).  Compare matfreens.py,
# from which it came.  Note that solve converges up to about Re=950.

from firedrake import *

N = 64
mesh = UnitSquareMesh(N, N)

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

Re = Constant(100.0)
F = ( (1.0 / Re) * inner(grad(u), grad(v)) * dx +
      inner(dot(grad(u), u), v) * dx -
      p * div(v) * dx +
      div(u) * q * dx )

bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

params = {"snes_monitor": None,
          "snes_converged_reason": None,
          "ksp_type": "preonly",
          "mat_type": "aij",
          "pc_type": "lu",
          "pc_factor_mat_solver_type": "mumps"}

print('solving by Newton with direct solve (MUMPS LU) for each step ...')
solve(F == 0, up, bcs=bcs, nullspace=nullspace,
      solver_parameters=params)

filename = 'cavity.pvd'
print('writing solution to %s ...' % filename)
u, p = up.split()
u.rename("velocity")
p.rename("pressure")
File(filename).write(u, p)

