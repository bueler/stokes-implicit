from firedrake import *

mesh = UnitSquareMesh(10,10)
x,y = SpatialCoordinate(mesh)

Vu = VectorFunctionSpace(mesh, 'CG', degree=2)
Vp = FunctionSpace(mesh, 'DG', degree=0)
Vc = FunctionSpace(mesh, 'CG', degree=1)
Z = Vu * Vp * Vc

upc = Function(Z)
u,p,c = split(upc)
v,q,e = TestFunctions(Z)

smb = - u[0] * y.dx(0) + u[1]
EquationBC(smb * e * ds(1) == 0, upc, 1, V=Z.sub(2))  # note last argument: specify test fcn space

