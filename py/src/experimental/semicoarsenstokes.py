# Rectangular lid-driven Stokes to test SemiCoarsenExtrudedHierarchy(), with
# a Schur fieldsplit solver using vertical-coarsening-only GMG on the velocity
# block.

from firedrake import *

N = 80
base = IntervalMesh(N,10.0)

# for X=3 the elements are square (aspect=1)
# X=3,4,5,6,7,8,9:  the number of iterations is constant and the time is linear as aspect goes to 64
X = 3
hierarchy = SemiCoarsenedExtrudedHierarchy(base, 1.0, base_layer=1, nref=X)
mesh = hierarchy[-1]

V = VectorFunctionSpace(mesh, 'CG', 2)
W = FunctionSpace(mesh, 'CG', 1)
Z = V * W

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

F = (inner(grad(u), grad(v)) - p * div(v) + div(u) * q - inner(Constant((0, 0)), v))*dx

bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), 'top'),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2)),
       DirichletBC(Z.sub(0), Constant((0, 0)), 'bottom')]

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

#boring direct solver if desired:
#params = {'snes_type': 'ksponly',
#          'ksp_type': 'preonly',
#          'mat_type': 'aij',
#          'pc_type': 'lu',
#          'pc_factor_mat_solver_type': 'mumps'}
#solve(F == 0, up, bcs=bcs, nullspace=nullspace, solver_parameters=params,
#      options_prefix='s')

params = {'snes_type': 'ksponly',
          'mat_type': 'nest',  # or 'aij'; both work and 'nest' slightly faster?
          'mat_type': 'aij',
          'ksp_type': 'gmres',
          'pc_type': 'fieldsplit',
          'pc_fieldsplit_type': 'schur',
          'pc_fieldsplit_schur_fact_type': 'lower',
          'pc_fieldsplit_schur_precondition': 'selfp',
          'fieldsplit_0_ksp_type': 'preonly',
          'fieldsplit_0_pc_type': 'mg',
          'fieldsplit_0_mg_coarse_ksp_type': 'preonly',
          'fieldsplit_0_mg_coarse_pc_type': 'lu',  # or 'gamg' if mat_type is aij
          #'fieldsplit_0_mg_coarse_ksp_type': 'gmres',
          #'fieldsplit_0_mg_coarse_ksp_max_it': 2,
          #'fieldsplit_0_mg_coarse_pc_type': 'ilu',
          'fieldsplit_1_ksp_type': 'preonly',
          #'fieldsplit_1_ksp_type': 'gmres',
          #'fieldsplit_1_ksp_rtol': 1.0e-1,
          'fieldsplit_1_pc_type': 'jacobi',
          'fieldsplit_1_pc_jacobi_type': 'diagonal'}
solve(F == 0, up, bcs=bcs, nullspace=nullspace, solver_parameters=params,
      options_prefix='s')

u, p = up.split()
u.rename('velocity')
p.rename('pressure')
File('scs.pvd').write(u,p)

