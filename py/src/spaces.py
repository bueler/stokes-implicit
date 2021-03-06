# generate finite element spaces for velocity, pressure, and displacement

import firedrake as fd

__all__ = ['vectorspaces']

# degrees of higher-order elements in the vertical
_degreexz = [(2,1),(3,2),(4,3),(5,4)]

def vectorspaces(mesh,vertical_higher_order=0):
    '''On an extruded mesh, build finite element spaces for velocity u,
    pressure p, and vertical displacement c.  Construct component spaces by
    explicitly applying TensorProductElement().  Elements are Q2 prisms
    [P2(triangle)xP2(interval)] for velocity, Q1 prisms [P1(triangle)xP1(interval)]
    for pressure, and Q1 prisms [P1(triangle)xP1(interval)] for displacement.
    Optionally the base mesh can be built from quadrilaterals and/or
    the vertical factors can be higher order for velocity and pressure.'''

    if mesh._base_mesh.cell_dimension() not in {1,2}:
        raise ValueError('only 2D and 3D extruded meshes are allowed')
    ThreeD = (mesh._base_mesh.cell_dimension() == 2)
    quad = (mesh._base_mesh.ufl_cell() == fd.quadrilateral)
    if quad and not ThreeD:
        raise ValueError('base mesh from quadilaterals only possible in 3D')
    zudeg,zpdeg = _degreexz[vertical_higher_order]

    # velocity u (vector)
    if ThreeD:
        if quad:
            xuE = fd.FiniteElement('Q',fd.quadrilateral,2)
        else:
            xuE = fd.FiniteElement('P',fd.triangle,2)
    else:
        xuE = fd.FiniteElement('P',fd.interval,2)
    zuE = fd.FiniteElement('P',fd.interval,zudeg)
    uE = fd.TensorProductElement(xuE,zuE)
    Vu = fd.VectorFunctionSpace(mesh,uE)

    # pressure p (scalar)
    #   Note Isaac et al (2015) recommend discontinuous pressure space
    #   to get mass conservation.  Switching base mesh elements to dP0
    #   (or dQ0 for base quads) is inconsistent w.r.t. velocity result;
    #   (unstable?) and definitely more expensive.
    if ThreeD:
        if quad:
            xpE = fd.FiniteElement('Q',fd.quadrilateral,1)
        else:
            xpE = fd.FiniteElement('P',fd.triangle,1)
    else:
        xpE = fd.FiniteElement('P',fd.interval,1)
    zpE = fd.FiniteElement('P',fd.interval,zpdeg)
    pE = fd.TensorProductElement(xpE,zpE)
    Vp = fd.FunctionSpace(mesh,pE)

    # vertical displacement c (scalar)
    if ThreeD:
        if quad:
            xcE = fd.FiniteElement('Q',fd.quadrilateral,1)
        else:
            xcE = fd.FiniteElement('P',fd.triangle,1)
    else:
        xcE = fd.FiniteElement('P',fd.interval,1)
    zcE = fd.FiniteElement('P',fd.interval,1)
    cE = fd.TensorProductElement(xcE,zcE)
    Vc = fd.FunctionSpace(mesh,cE)

    return Vu, Vp, Vc

