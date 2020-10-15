# Simplified 3D Navier-Stokes solver for lid-driven cavity using P2-P1
# Taylor-Hood elements and a direct solver (MUMPS).  Compare ns.py,

from firedrake import *

N = 20
mesh = UnitCubeMesh(N, N, N)

V = VectorFunctionSpace(mesh,'P',2)
W = FunctionSpace(mesh,'P',1)
Z = V * W

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

Re = Constant(100.0)
F = ( (1.0 / Re) * inner(grad(u), grad(v)) * dx +
      inner(dot(grad(u), u), v) * dx -
      p * div(v) * dx +
      div(u) * q * dx )

bcs = [DirichletBC(Z.sub(0), Constant((1, 1, 0)), (6,)),
       DirichletBC(Z.sub(0), Constant((0, 0, 0)), (1, 2, 3, 4, 5))]

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

filename = 'cavity3d.pvd'
print('writing solution to %s ...' % filename)
u, p = up.split()
u.rename("velocity")
p.rename("pressure")
File(filename).write(u, p)

