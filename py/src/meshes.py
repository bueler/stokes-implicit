# generate base and extruded meshes, including vertical stretching

import firedrake as fd

__all__ = ['basemesh']

def basemesh(L, mx, my=-1):
    '''Set up base mesh of intervals on [-L,L] if my<0 or quadilaterals on
    [-L,L]x[-L,L] otherwise.'''
    if my > 0:
        base_mesh = fd.RectangleMesh(mx, my, 2.0*L, 2.0*L, quadrilateral=True)
        base_mesh.coordinates.dat.data[:, 0] -= L
        base_mesh.coordinates.dat.data[:, 1] -= L
    else:
        base_mesh = fd.IntervalMesh(mx, length_or_left=0.0, right=2.0*L)
        base_mesh.coordinates.dat.data[:] -= L
    return base_mesh

