# generate base and extruded meshes, including vertical stretching

import firedrake as fd

__all__ = ['basemesh', 'extrudedmesh', 'extend', 'referencemesh']

def basemesh(L, mx, my=-1, quadrilateral=False):
    '''Set up base mesh of intervals on [-L,L] if my<0.  For 2D base mesh
    (my>0) use triangles, or optionally quadilaterals, on [-L,L]x[-L,L].'''
    if my > 0:
        base_mesh = fd.RectangleMesh(mx, my, 2.0*L, 2.0*L, quadrilateral=quadrilateral)
        base_mesh.coordinates.dat.data[:, 0] -= L
        base_mesh.coordinates.dat.data[:, 1] -= L
    else:
        base_mesh = fd.IntervalMesh(mx, length_or_left=0.0, right=2.0*L)
        base_mesh.coordinates.dat.data[:] -= L
    return base_mesh

def extrudedmesh(base_mesh, mz, refine=-1, temporary_height=1.0):
    '''Generate extruded mesh on draft reference domain.  Optional refinement
    hierarchy (if refine>0).  Returned mesh has placeholder height.'''
    if base_mesh.cell_dimension() not in {1,2}:
        raise ValueError('only 2D and 3D extruded meshes are generated')
    if refine > 0:
        hierarchy = fd.SemiCoarsenedExtrudedHierarchy(base_mesh, temporary_height,
                                                      base_layer=mz, nref=refine)
        mesh = hierarchy[-1]
        return mesh, hierarchy
    else:
        mesh = fd.ExtrudedMesh(base_mesh, layers=mz, layer_height=temporary_height/mz)
        return mesh

# on extruded mesh extend a function f(x,y), already defined on the base mesh,
# to the whole mesh using the 'R' constant-in-the-vertical space
def extend(mesh,f):
    if mesh._base_mesh.cell_dimension() == 2:
        if mesh._base_mesh.ufl_cell() == fd.quadrilateral:
            Q1R = fd.FunctionSpace(mesh,'Q',1,vfamily='R',vdegree=0)
        else:
            Q1R = fd.FunctionSpace(mesh,'P',1,vfamily='R',vdegree=0)
    elif mesh._base_mesh.cell_dimension() == 1:
        Q1R = fd.FunctionSpace(mesh,'P',1,vfamily='R',vdegree=0)
    else:
        raise ValueError('base mesh of extruded input mesh must be 1D or 2D')
    fextend = fd.Function(Q1R)
    fextend.dat.data[:] = f.dat.data_ro[:]
    return fextend

def _admissible(b, h):
    return all(h.dat.data >= b.dat.data)

def referencemesh(mesh, b, hinitial, Href):
    '''In-place modification of an extruded mesh to create the reference mesh.
    Changes the top surface to  lambda = b + max(Href,Hinitial - b).  Assumes
    the input mesh is extruded 3D mesh with  0 <= z <= 1.  Assumes b,hinitial
    are Functions defined on the base mesh.'''

    if not _admissible(b,hinitial):
        assert ValueError('input hinitial not admissible')
    P1base = fd.FunctionSpace(mesh._base_mesh,'P',1)
    Hstart = fd.Function(P1base).interpolate(hinitial - b)

    # FIXME  new form could be
    #   lam = referencemesh(mesh,b,hinitial,Href,tau)
    # which would do:
    #   2) get mesh resolution:  hT = min(mesh.cell_sizes.dat.data)
    #   3) set up k conditionally-stable time steps:
    #        Deltat = hT^2 / 4
    #        k = np.ceil(tau/Deltat)  # at least one; warn if >= 10
    #        Deltat = tau/k
    #   4) compute k steps of forward Euler heat equation  u_t = nabla^2 u
    #      with  u(t=0) = Hstart  and time steps  Deltat  ... yields Hend
    #   5) compute surface elevation of reference domain as at least Href:
    #        lambase = b + sqrt(Hend^2 + Href^2)
    lambase = fd.Function(P1base).interpolate(b + fd.max_value(Href, Hstart))

    lam = extend(mesh,lambase)
    Vcoord = mesh.coordinates.function_space()
    if mesh._base_mesh.cell_dimension() == 1:
        x,z = fd.SpatialCoordinate(mesh)
        f = fd.Function(Vcoord).interpolate(fd.as_vector([x,lam*z]))
    elif mesh._base_mesh.cell_dimension() == 2:
        x,y,z = fd.SpatialCoordinate(mesh)
        f = fd.Function(Vcoord).interpolate(fd.as_vector([x,y,lam*z]))
    else:
        raise ValueError('only 2D and 3D extruded meshes can be deformed')
    mesh.coordinates.assign(f)
    return 0

