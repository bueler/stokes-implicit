# module for defining UFL functionals for coupled weak form

import firedrake as fd
from .constants import g,rho,n,Bn  # for default values

__all__ = ['IceModel', 'IceModel2D']

class IceModel(object):
    '''Physics of the coupled ice flow and surface kinematical problems.'''

    def __init__(self, almost, mesh, Href, eps, Dtyp):
        self.almost = almost
        self.mesh = mesh
        self.Href = Href
        self.eps = eps
        self.Dtyp = Dtyp
        self.a = fd.Constant(0.0) # FIXME only correct for Halfar
        self.delta = 0.1

    def fbody(self):
        return fd.Constant((0.0, 0.0, - rho * g))  # 3D

    def _Falmost(self,u,p,c,v,q,e):
        '''This draft version uses the unmapped reference domain.'''
        Du = 0.5 * (fd.grad(u)+fd.grad(u).T)
        Dv = 0.5 * (fd.grad(v)+fd.grad(v).T)
        Du2 = 0.5 * fd.inner(Du, Du) + self.eps * self.Dtyp**2.0
        tau = Bn * Du2**(-1.0/n) * Du
        return fd.inner(tau, Dv) * fd.dx \
               + ( - p * fd.div(v) - fd.div(u) * q - fd.inner(self.fbody(),v) ) * fd.dx \
               + fd.inner(fd.grad(c),fd.grad(e)) * fd.dx

    def _j(self,c):  # 3D
        return fd.conditional(c.dx(2) >= self.delta, 1.0 + c.dx(2), self.delta)

    def _ell(self,c):
        return 1.0 / self._j(c)

    def _divmapped(self,u,c):  # 3D
        middle = c.dx(0) * u[0].dx(2) + c.dx(1) * u[1].dx(2)
        return fd.div(u) - self._ell(c) * middle + (self._ell(c) - 1.0) * u[2].dx(2)

    def _Dmapped(self,u,c):  # 3D
        term12 = 0.5 * (c.dx(1) * u[0].dx(2) + c.dx(0) * u[1].dx(2))
        term13 = 0.5 * c.dx(0) * u[2].dx(2)
        term23 = 0.5 * c.dx(1) * u[2].dx(2)
        Mcu = fd.as_tensor([[c.dx(0) * u[0].dx(2), term12,               term13],
                            [term12,               c.dx(1) * u[1].dx(2), term23],
                            [term13,               term23,               0.0   ]])
        Lu = fd.as_tensor([[0.0,              0.0,              0.5 * u[0].dx(2)],
                           [0.0,              0.0,              0.5 * u[1].dx(2)],
                           [0.5 * u[0].dx(2), 0.5 * u[1].dx(2), u[2].dx(2)      ]])
        return 0.5 * (fd.grad(u)+fd.grad(u).T) - self._ell(c) * Mcu + (self._ell(c) - 1.0) * Lu

    def _Ftrue(self,u,p,c,v,q,e):
        '''This version includes the  x -> xi  mapping.'''
        divu = self._divmapped(u,c)
        divv = self._divmapped(v,c)
        Du = self._Dmapped(u,c)
        Dv = self._Dmapped(v,c)
        Du2 = 0.5 * fd.inner(Du, Du) + self.eps * self.Dtyp**2.0
        #FIXME: MIASMA
        tau = Bn * Du2**(-1.0/n) * Du  # = 2 nu_e Du
        source = fd.inner(self.fbody(),v)
        return (fd.inner(tau, Dv) - p * divv - divu * q - source ) * self._j(c) * fd.dx \
               + fd.inner(fd.grad(c),fd.grad(e)) * fd.dx

    def F(self,u,p,c,v,q,e):
        '''Return the nonlinear weak form F(u,p,c;v,q,e) for the coupled
        Glen-Stokes and displacement problems.  Note the weak form for the
        top boundary condition is separate; see Fsmb().'''
        if self.almost:
            return self._Falmost(u,p,c,v,q,e)
        else:
            return self._Ftrue(u,p,c,v,q,e)

    def zcoord(self,mesh):  # 3D
        _,_,z = fd.SpatialCoordinate(mesh)
        return z

    def tangentu(self,u,z):  # 3D
        return u[0] * z.dx(0) + u[1] * z.dx(1)

    def vertu(self,u):  # 3D
        return u[2]

    def smbref(self,dt,z,smb):
        return fd.conditional(z > self.Href, dt * smb, dt * smb - self.Href)

    def Fsmb(self,mesh,Z,dt,u,c,e):
        '''Return the weak form Fsmb(c;e) of the top boundary condition
        for the displacement problem so we may apply the surface kinematical
        equation weakly.  This weak form also depends on u.'''
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

    def fbody(self):
        return fd.Constant((0.0, - rho * g))

    def _j(self,c):
        return fd.conditional(c.dx(1) >= self.delta, 1.0 + c.dx(1), self.delta)

    def _divmapped(self,u,c):
        middle = c.dx(0) * u[0].dx(1)
        return fd.div(u) - self._ell(c) * middle + (self._ell(c) - 1.0) * u[1].dx(1)

    def _Dmapped(self,u,c):
        term12 = 0.5 * c.dx(0) * u[1].dx(1)
        Mcu = fd.as_tensor([[c.dx(0) * u[0].dx(1), term12],
                            [term12,               0.0   ]])
        Lu = fd.as_tensor([[0.0,              0.5 * u[0].dx(1)],
                           [0.5 * u[0].dx(1), u[1].dx(1)      ]])
        return 0.5 * (fd.grad(u)+fd.grad(u).T) - self._ell(c) * Mcu + (self._ell(c) - 1.0) * Lu

    def zcoord(self,mesh):
        _,z = fd.SpatialCoordinate(mesh)
        return z

    def tangentu(self,u,z):
        return u[0] * z.dx(0)

    def vertu(self,u):
        return u[1]

