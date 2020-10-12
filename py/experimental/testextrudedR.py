from firedrake import *

base_mesh = UnitIntervalMesh(10)
mesh = ExtrudedMesh(base_mesh, layers=5, layer_height=1.0)
xpE = FiniteElement('CG',interval,1)
zrE = FiniteElement('R',interval)
constantE = TensorProductElement(xpE,zrE)
constantV = FunctionSpace(mesh,constantE)

