# generate and save diagnostic quantities on extruded meshes

import firedrake as fd
from .constants import g,rho,n,Bn,Gamma

__all__ = ['stresses', 'surfaceelevation', 'extendsurfaceelevation',
           'pdifference', 'siahorizontalvelocity', 'writeresult']

# regularized tensor-valued deviatoric stress tau and effective viscosity from the velocity solution
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

# compute jweight function
def jweight(mesh,icemodel,c):
    Q1 = fd.FunctionSpace(mesh,'Q',1)
    jweight = fd.Function(Q1).interpolate(icemodel.jweight(c))
    jweight.rename('jweight')
    return jweight

# compute h(x,y), defined on the base mesh, from the top coordinate of the extruded mesh
def surfaceelevation(mesh):
    Q1 = fd.FunctionSpace(mesh,'Q',1)
    if mesh._base_mesh.cell_dimension() == 2:
        Q1base = fd.FunctionSpace(mesh._base_mesh,'Q',1)
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

# extend a function f(x,y) to extruded mesh using the 'R' constant-in-the-vertical space
def extend(mesh,f):
    if mesh._base_mesh.cell_dimension() == 2:
        Q1R = fd.FunctionSpace(mesh,'Q',1,vfamily='R',vdegree=0)
    elif mesh._base_mesh.cell_dimension() == 1:
        Q1R = fd.FunctionSpace(mesh,'P',1,vfamily='R',vdegree=0)
    else:
        raise ValueError('base mesh of extruded input mesh must be 1D or 2D')
    fextend = fd.Function(Q1R)
    fextend.dat.data[:] = f.dat.data_ro[:]
    return fextend

# hydrostatic pressure
def phydrostatic(mesh):
    Q1 = fd.FunctionSpace(mesh,'Q',1)
    hbase = surfaceelevation(mesh)
    h = extend(mesh,hbase)
    x = fd.SpatialCoordinate(mesh)
    z = x[mesh._base_mesh.cell_dimension()]
    ph = fd.Function(Q1).interpolate(rho * g * (h - z))
    ph.rename('hydrostatic pressure')
    return ph

# density as a piecewise constant
def densityicemiasma(mesh,hcurrentextruded,rhom):
    x = fd.SpatialCoordinate(mesh)
    z = x[mesh._base_mesh.cell_dimension()]
    Q0 = fd.FunctionSpace(mesh,'DQ',0)
    rhofield = fd.Function(Q0).project(fd.conditional(z < hcurrentextruded, rho, rhom))
    rhofield.rename('density ice/miasma')
    return rhofield

# horizontal velocity <u,v> from the shallow ice approximation (SIA)
def siahorizontalvelocity(mesh):
    hbase = surfaceelevation(mesh)
    if mesh._base_mesh.cell_dimension() == 2:
        Vvectorbase = fd.VectorFunctionSpace(mesh._base_mesh,'DQ',0)
        gradhbase = fd.project(fd.grad(hbase),Vvectorbase)
        VvectorR = fd.VectorFunctionSpace(mesh,'DQ',0, vfamily='R', vdegree=0, dim=2)
        Vvector = fd.VectorFunctionSpace(mesh,'DQ',0, dim=2)
    elif mesh._base_mesh.cell_dimension() == 1:
        Vvectorbase = fd.FunctionSpace(mesh._base_mesh,'DP',0)
        gradhbase = fd.project(hbase.dx(0),Vvectorbase)
        VvectorR = fd.FunctionSpace(mesh,'DP',0, vfamily='R', vdegree=0)
        Vvector = fd.FunctionSpace(mesh,'DP',0)
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

# save ParaView-readable file
def writeresult(filename,mesh,icemodel,upc,hinitialextruded,saveextra=False):
    assert filename.split('.')[-1] == 'pvd'
    written = 'u,p,c'
    if mesh.comm.size > 1:
         written += ',rank'
    if saveextra:
         written += ',tau,nu,rho,phydrostatic,jweight,velocitySIA'
    fd.PETSc.Sys.Print('writing solution variables (%s) to output file %s ... ' \
                       % (written,filename))
    u,p,c = upc.split()
    u.rename('velocity')
    p.rename('pressure')
    c.rename('displacement')
    if saveextra:
        tau, nu = stresses(mesh,icemodel,u)
        rhofield = densityicemiasma(mesh,hinitialextruded,rho / 10.0)
        ph = phydrostatic(mesh)
        jw = jweight(mesh,icemodel,c)
        velocitySIA = siahorizontalvelocity(mesh)
    if mesh.comm.size > 1:
        # integer-valued element-wise process rank
        rank = fd.Function(fd.FunctionSpace(mesh,'DG',0))
        rank.dat.data[:] = mesh.comm.rank
        rank.rename('rank')
        if saveextra:
            fd.File(filename).write(u,p,c,rank,tau,nu,rhofield,ph,jw,velocitySIA)
        else:
            fd.File(filename).write(u,p,c,rank)
    else:
        if saveextra:
            fd.File(filename).write(u,p,c,tau,nu,rhofield,ph,jw,velocitySIA)
        else:
            fd.File(filename).write(u,p,c)

