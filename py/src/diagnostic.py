# generate and save diagnostic quantities on extruded meshes

import firedrake as fd
from .constants import g,rho,n,Bn,Gamma
from .meshes import extend, extrudedmesh
from .spaces import vectorspaces

__all__ = ['stresses', 'jweight', 'surfaceelevation',
           'phydrostatic', 'siahorizontalvelocity',
           'writereferenceresult', 'writesolutiongeometry']

# on mesh get regularized tensor-valued deviatoric stress tau
# and effective viscosity from the velocity solution
def stresses(mesh,icemodel,u):
    Q1 = fd.FunctionSpace(mesh,'Q',1)
    TQ1 = fd.TensorFunctionSpace(mesh,'Q',1)
    Du = fd.Function(TQ1).interpolate(0.5 * (fd.grad(u)+fd.grad(u).T))
    Du2 = fd.Function(Q1).interpolate(0.5 * fd.inner(Du, Du) + icemodel.eps * icemodel.Dtyp**2.0)
    nu = fd.Function(Q1).interpolate(0.5 * Bn * Du2**(-1.0/n))
    nu.rename('effective viscosity')
    tau = fd.Function(TQ1).interpolate(2.0 * nu * Du)
    tau.rename('tau')
    return tau, nu

# on mesh compute jweight function
def jweight(mesh,icemodel,c):
    Q1 = fd.FunctionSpace(mesh,'Q',1)
    jweight = fd.Function(Q1).interpolate(icemodel.jweight(c))
    jweight.rename('jweight')
    return jweight

# on extruded mesh extract the value along the top surface as a P1 or Q1
# function on the base mesh
def _surfacevalue(mesh,f):
    if mesh._base_mesh.cell_dimension() == 2:
        if mesh._base_mesh.ufl_cell() == fd.quadrilateral:
            Q1base = fd.FunctionSpace(mesh._base_mesh,'Q',1)
        else:
            Q1base = fd.FunctionSpace(mesh._base_mesh,'P',1)
    elif mesh._base_mesh.cell_dimension() == 1:
        Q1base = fd.FunctionSpace(mesh._base_mesh,'P',1)
    else:
        raise ValueError('base mesh of extruded input mesh must be 1D or 2D')
    fbase = fd.Function(Q1base)
    Q1 = fd.FunctionSpace(mesh,'Q',1)
    bc = fd.DirichletBC(Q1,1.0,'top')
    # add halos for parallel interpolation
    fbase.dat.data_with_halos[:] = f.dat.data_with_halos[bc.nodes]
    return fbase, Q1base

# on extruded mesh compute the surface elevation h(x,y), defined as a
# P1 or Q1 function on the base mesh, from the top coordinate of the mesh
def surfaceelevation(mesh):
    x = fd.SpatialCoordinate(mesh)
    # z itself is an 'Indexed' object, so use a Function with a .dat attribute
    Q1 = fd.FunctionSpace(mesh,'Q',1)
    z = fd.Function(Q1).interpolate(x[mesh._base_mesh.cell_dimension()])
    hbase,_ = _surfacevalue(mesh,z)
    return hbase

# on extruded mesh compute hydrostatic pressure
def phydrostatic(mesh):
    Q1 = fd.FunctionSpace(mesh,'Q',1)
    hbase = surfaceelevation(mesh)
    h = extend(mesh,hbase)
    x = fd.SpatialCoordinate(mesh)
    z = x[mesh._base_mesh.cell_dimension()]
    ph = fd.Function(Q1).interpolate(rho * g * (h - z))
    ph.rename('hydrostatic pressure')
    return ph

# on extruded mesh compute the horizontal velocity <u,v> in the shallow
# ice approximation (SIA)
def siahorizontalvelocity(mesh):
    hbase = surfaceelevation(mesh)
    if mesh._base_mesh.cell_dimension() == 2:
        if mesh._base_mesh.ufl_cell() == fd.quadrilateral:
            Vvectorbase = fd.VectorFunctionSpace(mesh._base_mesh,'DQ',0)
            VvectorR = fd.VectorFunctionSpace(mesh,'DQ',0, vfamily='R', vdegree=0, dim=2)
        else:
            Vvectorbase = fd.VectorFunctionSpace(mesh._base_mesh,'DP',0)
            VvectorR = fd.VectorFunctionSpace(mesh,'DP',0, vfamily='R', vdegree=0, dim=2)
        gradhbase = fd.project(fd.grad(hbase),Vvectorbase)
        Vvector = fd.VectorFunctionSpace(mesh,'DQ',0, dim=2)
    elif mesh._base_mesh.cell_dimension() == 1:
        Vvectorbase = fd.FunctionSpace(mesh._base_mesh,'DP',0)
        gradhbase = fd.project(hbase.dx(0),Vvectorbase)
        VvectorR = fd.FunctionSpace(mesh,'DP',0, vfamily='R', vdegree=0)
        Vvector = fd.FunctionSpace(mesh,'DQ',0)
    else:
        raise ValueError('base mesh of extruded input mesh must be 1D or 2D')
    gradh = fd.Function(VvectorR)
    gradh.dat.data[:] = gradhbase.dat.data_ro[:]
    h = extend(mesh,hbase)
    DQ0 = fd.FunctionSpace(mesh,'DQ',0)
    h0 = fd.project(h,DQ0)
    x = fd.SpatialCoordinate(mesh)
    z0 = fd.project(x[mesh._base_mesh.cell_dimension()],DQ0)
    # FIXME following only valid in flat bed case
    uvsia = - Gamma * (h0**(n+1) - (h0-z0)**(n+1)) * abs(gradh)**(n-1) * gradh
    uv = fd.Function(Vvector).interpolate(uvsia)
    uv.rename('velocitySIA')
    return uv

# save ParaView-readable file for reference domain
# this file shows full computational context, and diagnostics, and could be
# re-started from
def writereferenceresult(filename,mesh,icemodel,upc):
    assert filename.split('.')[-1] == 'pvd'
    variables = 'u,p,c,tau,nu,phydrostatic,jweight,velocitySIA'
    if mesh.comm.size > 1:
         variables += ',rank'
    fd.PETSc.Sys.Print('writing variables (%s) to output file %s ... ' \
                       % (variables,filename))
    u,p,c = upc.split()
    u.rename('velocity')
    p.rename('pressure')
    c.rename('displacement')
    tau, nu = stresses(mesh,icemodel,u)
    ph = phydrostatic(mesh)
    jw = jweight(mesh,icemodel,c)
    velocitySIA = siahorizontalvelocity(mesh)
    if mesh.comm.size > 1:
        # integer-valued element-wise process rank
        rank = fd.Function(fd.FunctionSpace(mesh,'DG',0))
        rank.dat.data[:] = mesh.comm.rank
        rank.rename('rank')
        fd.File(filename).write(u,p,c,tau,nu,ph,jw,velocitySIA,rank)
    else:
        fd.File(filename).write(u,p,c,tau,nu,ph,jw,velocitySIA)
    return 0

# save Paraview-readable file with new solution geometry and u,p
# (velocity,pressure) solution; note mesh is crushed in ice-free areas
# FIXME currently writes inadmissible (i.e. negative) z values
def writesolutiongeometry(filename,refmesh,mzfine,upc):
    # get fields on reference mesh
    u,p,c = upc.split()

    # compute hbase as updated surface elevation
    #   h(x,y) = lambda(x,y) + c(x,y,lambda(x,y))
    lambase = surfaceelevation(refmesh)  # bounded below by Href; space Q1base
    cbase, Q1base = _surfacevalue(refmesh,c)
    hbase = fd.Function(Q1base).interpolate(lambase + cbase)

    # duplicate fine mesh and change its coordinate to linear times h
    mesh = extrudedmesh(refmesh._base_mesh,mzfine,refine=-1,temporary_height=1.0)
    h = extend(mesh,hbase)
    Vcoord = mesh.coordinates.function_space()
    if mesh._base_mesh.cell_dimension() == 1:
        x,z = fd.SpatialCoordinate(mesh)
        Xnew = fd.Function(Vcoord).interpolate(fd.as_vector([x,h*z]))
    elif mesh._base_mesh.cell_dimension() == 2:
        x,y,z = fd.SpatialCoordinate(mesh)
        Xnew = fd.Function(Vcoord).interpolate(fd.as_vector([x,y,h*z]))
    else:
        raise ValueError('applies only to 2D and 3D extruded meshes')
    mesh.coordinates.assign(Xnew)

    # interpolate velocity u and pressure p from refmesh onto mesh and write
    Vu,Vp,_ = vectorspaces(mesh)
    unew = fd.Function(Vu)
    pnew = fd.Function(Vp)
    unew.rename('velocity')
    pnew.rename('pressure')
    # note f.at() searches for element to evaluate (thanks L Mitchell)
    print(Xnew.dat.data_ro)  # FIXME shows inadmissible z
    unew.dat.data[:] = u.at(Xnew.dat.data_ro)
    pnew.dat.data[:] = p.at(Xnew.dat.data_ro)
    fd.File(filename).write(unew,pnew)
    return 0

