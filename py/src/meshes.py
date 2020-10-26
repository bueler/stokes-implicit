# generate base and extruded meshes, including vertical stretching

import firedrake as fd

__all__ = ['basemesh', 'extrudedmesh', 'deformlimitmesh']

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

def extrudedmesh(base_mesh, mz, refine=-1, temporary_height=1.0):
    '''Generate extruded mesh on reference domain, optionally with refinement
    hierarchy (if refine>0).  Returned mesh has placeholder height.'''
    dim = base_mesh.cell_dimension() + 1
    if dim not in {2,3}:
        raise ValueError('only 2D and 3D extruded meshes are generated')
    if refine > 0:
        hierarchy = fd.SemiCoarsenedExtrudedHierarchy(base_mesh, temporary_height,
                                                      base_layer=mz, nref=refine)
        mesh = hierarchy[-1]
        return mesh, hierarchy
    else:
        mesh = fd.ExtrudedMesh(base_mesh, layers=mz, layer_height=temporary_height/mz)
        return mesh

def deformlimitmesh(mesh, Hinitial, Href):
    '''Modify an extruded mesh: Change vertical coordinate to max(Hinitial,Href).'''
    # FIXME allow for nonzero bed elevation
    Hlimited = fd.max_value(Href, Hinitial)
    Vcoord = mesh.coordinates.function_space()
    if mesh._base_mesh.cell_dimension() == 1:
        x,z = fd.SpatialCoordinate(mesh)
        f = fd.Function(Vcoord).interpolate(fd.as_vector([x,Hlimited*z]))
    elif mesh._base_mesh.cell_dimension() == 2:
        x,y,z = fd.SpatialCoordinate(mesh)
        f = fd.Function(Vcoord).interpolate(fd.as_vector([x,y,Hlimited*z]))
    else:
        raise ValueError('only 2D and 3D extruded meshes can be deformed')
    mesh.coordinates.assign(f)
    return 0

