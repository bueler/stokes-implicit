# generate base and extruded meshes, including vertical stretching

import firedrake as fd

__all__ = ['basemesh', 'extrudedmesh', 'referencemesh']

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

def referencemesh(mesh, b, hinitial, Href):
    '''In-place modification of an extruded mesh to create the reference mesh.
    Changes the top surface to  lambda = b + max(Href,Hinitial - b).  Assumes
    the input mesh is extruded 3D mesh with  0 <= z <= 1.'''
    # FIXME  new form could be
    #   lam = referencemesh(mesh,b,hinitial,Href,tau)
    # which would do:
    #   1) compute thickness and check admissible:  Hstart = hinitial - b >= 0
    #   2) get mesh resolution:  hT = min(mesh.cell_sizes.dat.data)
    #   3) set up k conditionally-stable time steps:
    #        Deltat = hT^2 / 4
    #        k = np.ceil(tau/Deltat)  # at least one; warn if >= 10
    #        Deltat = tau/k
    #   4) compute k steps of forward Euler heat equation  u_t = nabla^2 u
    #      with  u(t=0) = Hstart  and time steps  Deltat  ... yields Hend
    #   5) compute the new surface elevation so that it is at least Href:
    #        lam = b + sqrt(Hend^2 + Href^2)
    hlimited = b + fd.max_value(Href, hinitial - b)
    Vcoord = mesh.coordinates.function_space()
    if mesh._base_mesh.cell_dimension() == 1:
        x,z = fd.SpatialCoordinate(mesh)
        f = fd.Function(Vcoord).interpolate(fd.as_vector([x,hlimited*z]))
    elif mesh._base_mesh.cell_dimension() == 2:
        x,y,z = fd.SpatialCoordinate(mesh)
        f = fd.Function(Vcoord).interpolate(fd.as_vector([x,y,hlimited*z]))
    else:
        raise ValueError('only 2D and 3D extruded meshes can be deformed')
    mesh.coordinates.assign(f)
    return 0

