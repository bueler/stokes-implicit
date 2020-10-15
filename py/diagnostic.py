# glacier modeling diagnostic calculations on extruded meshes

from firedrake import *
from iceconstants import g,rho,n,Bn,Gamma

# regularized tensor-valued deviatoric stress tau and effective viscosity from the velocity solution
def stresses(mesh,u,eps,Dtyp):
    Q1 = FunctionSpace(mesh,'Q',1)
    TQ1 = TensorFunctionSpace(mesh,'Q',1)
    Du = Function(TQ1).interpolate(0.5 * (grad(u)+grad(u).T))
    Du2 = Function(Q1).interpolate(0.5 * inner(Du, Du) + eps * Dtyp**2.0)
    nu = Function(Q1).interpolate(0.5 * Bn * Du2**(-1.0/n))
    nu.rename('effective viscosity')
    tau = Function(TQ1).interpolate(2.0 * nu * Du)
    tau.rename('tau')
    return tau, nu

# h(x,y) on the base mesh
def surfaceelevation(mesh):
    Q1 = FunctionSpace(mesh,'Q',1)
    if mesh._base_mesh.cell_dimension() == 2:
        Q1base = FunctionSpace(mesh._base_mesh,'Q',1)
    elif mesh._base_mesh.cell_dimension() == 1:
        Q1base = FunctionSpace(mesh._base_mesh,'P',1)
    else:
        raise ValueError('base mesh of extruded input mesh must be 1D or 2D')
    hbase = Function(Q1base)
    bc = DirichletBC(Q1,1.0,'top')
    x = SpatialCoordinate(mesh)
    # z itself is an 'Indexed' object, so use a Function with a .dat attribute
    z = Function(Q1).interpolate(x[mesh._base_mesh.cell_dimension()])
    # add halos for parallelizability of the interpolation
    hbase.dat.data_with_halos[:] = z.dat.data_with_halos[bc.nodes]
    return hbase

# extend h(x,y) to extruded mesh using the 'R' constant-in-the-vertical space
def extendsurfaceelevation(mesh):
    hbase = surfaceelevation(mesh)
    if mesh._base_mesh.cell_dimension() == 2:
        Q1R = FunctionSpace(mesh,'Q',1,vfamily='R',vdegree=0)
    elif mesh._base_mesh.cell_dimension() == 1:
        Q1R = FunctionSpace(mesh,'P',1,vfamily='R',vdegree=0)
    else:
        raise ValueError('base mesh of extruded input mesh must be 1D or 2D')
    h = Function(Q1R)
    h.dat.data[:] = hbase.dat.data_ro[:]
    return h

# difference between pressure and hydrostatic pressure
def pdifference(mesh,p):
    Q1 = FunctionSpace(mesh,'Q',1)
    h = extendsurfaceelevation(mesh)
    x = SpatialCoordinate(mesh)
    pdiff = Function(Q1).interpolate(p - rho * g * (h - x[mesh._base_mesh.cell_dimension()]))
    pdiff.rename('pressure minus hydrostatic pressure')
    return pdiff

# horizontal velocity <u,v> from the shallow ice approximation (SIA)
def siahorizontalvelocity(mesh):
    hbase = surfaceelevation(mesh)
    if mesh._base_mesh.cell_dimension() == 2:
        Vvectorbase = VectorFunctionSpace(mesh._base_mesh,'DQ',0)
        gradhbase = project(grad(hbase),Vvectorbase)
        VvectorR = VectorFunctionSpace(mesh,'DQ',0, vfamily='R', vdegree=0, dim=2)
        Vvector = VectorFunctionSpace(mesh,'DQ',0, dim=2)
    elif mesh._base_mesh.cell_dimension() == 1:
        Vvectorbase = FunctionSpace(mesh._base_mesh,'DP',0)
        gradhbase = project(hbase.dx(0),Vvectorbase)
        VvectorR = FunctionSpace(mesh,'DP',0, vfamily='R', vdegree=0)
        Vvector = FunctionSpace(mesh,'DP',0)
    else:
        raise ValueError('base mesh of extruded input mesh must be 1D or 2D')
    gradh = Function(VvectorR)
    gradh.dat.data[:] = gradhbase.dat.data_ro[:]
    h = extendsurfaceelevation(mesh)
    DQ0 = FunctionSpace(mesh,'DQ',0)
    h0 = project(h,DQ0)
    x = SpatialCoordinate(mesh)
    z0 = project(x[mesh._base_mesh.cell_dimension()],DQ0)
    # FIXME following only valid in flat bed case
    uv = Function(Vvector).interpolate(- Gamma * (h0**(n+1) - (h0-z0)**(n+1)) * abs(gradh)**(n-1) * gradh)
    uv.rename('velocitySIA')
    return uv

