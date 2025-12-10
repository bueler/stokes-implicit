from firedrake import *
from geometryinit import R0, H0,generategeometry

def test_geometryinit():
    L = 1.3 * R0
    basemesh = RectangleMesh(6, 6, L, L, originX=-L, originY=-L)
    x = SpatialCoordinate(basemesh)
    V = FunctionSpace(basemesh, "CG", 1)
    b, s = generategeometry(V, x)
    with s.dat.vec_ro as sv:
        smax = sv.max()[1]
        smin = sv.min()[1]
    assert smax == H0
    assert smin < -100.0

if __name__ == "__main__":
    pass
    test_geometryinit()
