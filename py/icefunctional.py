# module for defining UFL functionals for coupled weak form

from iceconstants import g,rho,n,Bn  # for default values
import firedrake as fd

class IceModel(object):
    '''Physics of the coupled ice flow and surface kinematical problem
    in the 3D case.'''

    def __init__(self, mesh, Href, eps, Dtyp):
        self.mesh = mesh
        self.Href = Href
        self.eps = eps
        self.Dtyp = Dtyp
        self.a = fd.Constant(0.0) # FIXME only correct for Halfar
        self.f_body = fd.Constant((0.0, 0.0, - rho * g))

    def F(self,u,p,c,v,q,e):
        '''Return the nonlinear weak form F(u,p,c;v,q,e), without the top
        boundary term, for coupled Glen law Stokes and displacement problems.'''
        # FIXME need stretching in F; couples (u,p) and c problems
        Du = 0.5 * (fd.grad(u)+fd.grad(u).T)
        Dv = 0.5 * (fd.grad(v)+fd.grad(v).T)
        Du2 = 0.5 * fd.inner(Du, Du) + self.eps * self.Dtyp**2.0
        tau = Bn * Du2**(-1.0/n) * Du
        return fd.inner(tau, Dv) * fd.dx \
               + ( - p * fd.div(v) - fd.div(u) * q - fd.inner(self.f_body,v) ) * fd.dx \
               + fd.inner(fd.grad(c),fd.grad(e)) * fd.dx

    def zcoord(self,mesh):
        _,_,z = fd.SpatialCoordinate(mesh)
        return z

    def tangentu(self,u,z):
        return u[0] * z.dx(0) + u[1] * z.dx(1)

    def vertu(self,u):
        return u[2]

    def smbref(self,dt,z,smb):
        return fd.conditional(z > self.Href, dt * smb, dt * smb - self.Href)

    def Fsmb(self,mesh,Z,dt,u,c,e):
        '''Return the weak form Fsmb(u;c;e) of the top boundary condition
        for the displacement problem so we may apply the surface kinematical
        equation weakly.'''
        z = self.zcoord(mesh)
        smb = self.a - self.tangentu(u,z) + self.vertu(u)
        return (c - self.smbref(dt,z,smb)) * e * fd.ds_t

    def Dirichletsmb(self,mesh,dt):
        '''Returns a DirichletBC if we are NOT applying the surface kinematical
        equation weakly.'''
        z = self.zcoord(mesh)
        return self.smbref(dt,z,self.a)

class IceModel2D(IceModel):
    '''Physics of the coupled ice flow and surface kinematical problem
    in the 2D (flowline) case.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f_body = fd.Constant((0.0, - rho * g))

    def zcoord(self,mesh):
        _,z = fd.SpatialCoordinate(mesh)
        return z

    def tangentu(self,u,z):
        return u[0] * z.dx(0)

    def vertu(self,u):
        return u[1]

