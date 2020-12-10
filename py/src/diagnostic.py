# generate and save diagnostic quantities on extruded meshes

import firedrake as fd
from .constants import g,rho,n,Bn,Gamma
from .meshes import extend

__all__ = ['stresses', 'jweight', 'surfaceelevation',
           'phydrostatic', 'siahorizontalvelocity',
           'writereferenceresult']

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

# on extruded mesh compute the surface elevation h(x,y), defined on the base
# mesh, from the top coordinate of the mesh
def surfaceelevation(mesh):
    Q1 = fd.FunctionSpace(mesh,'Q',1)
    if mesh._base_mesh.cell_dimension() == 2:
        if mesh._base_mesh.ufl_cell() == fd.quadrilateral:
            Q1base = fd.FunctionSpace(mesh._base_mesh,'Q',1)
        else:
            Q1base = fd.FunctionSpace(mesh._base_mesh,'P',1)
    elif mesh._base_mesh.cell_dimension() == 1:
        Q1base = fd.FunctionSpace(mesh._base_mesh,'P',1)
    else:
        raise ValueError('base mesh of extruded input mesh must be 1D or 2D')
    hbase = fd.Function(Q1base)
    bc = fd.DirichletBC(Q1,1.0,'top')
    x = fd.SpatialCoordinate(mesh)
    # z itself is an 'Indexed' object, so use a Function with a .dat attribute
    z = fd.Function(Q1).interpolate(x[mesh._base_mesh.cell_dimension()])
    # add halos for parallelizability of the interpolation
    hbase.dat.data_with_halos[:] = z.dat.data_with_halos[bc.nodes]
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

# FIXME how to do step 3 below?
#   1. from c in solution, compute new surface elevation by
#        h(x,y) = lambda(x,y) + c(x,y,lambda(x,y))
#   2. generate a new mesh by copying old and then using h to define vertical
#      (which will be zero in ice-free regions)
#   3. interpolate values of velocity u and pressure p to this new mesh
#   4. write u,p on new mesh
#def writesolutiongeometry(filename,mesh,icemodel,upc)

