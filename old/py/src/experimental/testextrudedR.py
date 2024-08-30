from firedrake import *

base_mesh = UnitIntervalMesh(10)
mesh = ExtrudedMesh(base_mesh, layers=5, layer_height=1.0)

Q2D = FunctionSpace(base_mesh, 'CG', 1)
Q3D = FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)

f2D = Function(Q2D)
f3D = Function(Q3D)
f3D.dat.data[:] = f2D.dat.data_ro[:]

