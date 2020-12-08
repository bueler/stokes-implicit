# generate finite element spaces for velocity, pressure, and displacement

import firedrake as fd

__all__ = ['vectorspaces']

# degrees of higher-order elements in the vertical
_degreexz = [(2,1),(3,2),(4,3),(5,4)]

def vectorspaces(mesh,vertical_higher_order=0):
    '''On an extruded mesh, build finite element spaces for velocity u,
    pressure p, and vertical displacement c.  Construct component spaces by
    explicitly applying TensorProductElement().'''

    if mesh._base_mesh.cell_dimension() not in {1,2}:
        raise ValueError('only 2D and 3D extruded meshes are allowed')
    ThreeD = mesh._base_mesh.cell_dimension() == 2
    zudeg,zpdeg = _degreexz[vertical_higher_order]

    # velocity u (vector)
    if ThreeD:
        xuE = fd.FiniteElement('Q',fd.quadrilateral,2)
    else:
        xuE = fd.FiniteElement('P',fd.interval,2)
    zuE = fd.FiniteElement('P',fd.interval,zudeg)
    uE = fd.TensorProductElement(xuE,zuE)
    Vu = fd.VectorFunctionSpace(mesh, uE)

    # pressure p (scalar)
    #   note Isaac et al (2015) recommend discontinuous pressure space
    #   to get mass conservation but using dQ0 seems unstable and dQ1
    #   notably more expensive
    if ThreeD:
        xpE = fd.FiniteElement('Q',fd.quadrilateral,1)
    else:
        xpE = fd.FiniteElement('P',fd.interval,1)
    zpE = fd.FiniteElement('P',fd.interval,zpdeg)
    pE = fd.TensorProductElement(xpE,zpE)
    Vp = fd.FunctionSpace(mesh, pE)

    # vertical displacement c (scalar)
    if ThreeD:
        xcE = fd.FiniteElement('Q',fd.quadrilateral,1)
    else:
        xcE = fd.FiniteElement('P',fd.interval,1)
    zcE = fd.FiniteElement('P',fd.interval,1)
    cE = fd.TensorProductElement(xcE,zcE)
    Vc = fd.FunctionSpace(mesh, cE)

    return Vu, Vp, Vc

