# module for defining UFL functionals for coupled weak form

import firedrake as fd
from .constants import g,rho,n,Bn  # for default values

__all__ = ['IceModel', 'IceModel2D']

class IceModel(object):
    '''Physics of the coupled ice flow and surface kinematical problems.'''

    def __init__(self, mesh=None, almost=None, Href=None, eps=None, Dtyp=None, hcurrent=None, rhom=None):
        self.almost = almost
        self.mesh = mesh
        self.Href = Href
        self.eps = eps
        self.Dtyp = Dtyp
        self.hcurrent = hcurrent
        self.rhom = rhom
        self.delta = 0.1
        self.qdegree = 3  # used in mapped weak form FIXME how to determine a wise value?

    def _fbody(self, rhofield):  # 3D
        return fd.as_vector([fd.Constant(0.0), fd.Constant(0.0), - rhofield * fd.Constant(g)])

    def _Falmost(self,u,p,c,v,q,e):
        '''This draft version uses the unmapped reference domain.'''
        Du = 0.5 * (fd.grad(u)+fd.grad(u).T)
        Dv = 0.5 * (fd.grad(v)+fd.grad(v).T)
        Du2 = 0.5 * fd.inner(Du, Du) + self.eps * self.Dtyp**2.0
        tau = Bn * Du2**(-1.0/n) * Du
        source = fd.inner(self._fbody(fd.Constant(rho)),v)
        return fd.inner(tau, Dv) * fd.dx \
               + ( - p * fd.div(v) - fd.div(u) * q - source ) * fd.dx \
               + fd.inner(fd.grad(c),fd.grad(e)) * fd.dx

    def _czeta(self,c):  # 3D
        return c.dx(2)

    def _divmiddle(self,u,c):  # 3D
        return c.dx(0) * u[0].dx(2) + c.dx(1) * u[1].dx(2)

    def _w(self,u):  # 3D
        return u[2]

    def _wzeta(self,u):  # 3D
        return u[2].dx(2)

    def _Mcu(self,u,c):  # 3D
        term12 = 0.5 * (c.dx(1) * u[0].dx(2) + c.dx(0) * u[1].dx(2))
        term13 = 0.5 * c.dx(0) * u[2].dx(2)
        term23 = 0.5 * c.dx(1) * u[2].dx(2)
        return fd.as_tensor([[c.dx(0) * u[0].dx(2), term12,               term13],
                             [term12,               c.dx(1) * u[1].dx(2), term23],
                             [term13,               term23,               0.0   ]])

    def _Lu(self,u,c):  # 3D
        return fd.as_tensor([[0.0,              0.0,              0.5 * u[0].dx(2)],
                             [0.0,              0.0,              0.5 * u[1].dx(2)],
                             [0.5 * u[0].dx(2), 0.5 * u[1].dx(2), u[2].dx(2)      ]])

    def jweight(self,c):
        return fd.max_value(1.0 + self._czeta(c), self.delta)

    def _ell(self,c):
        return 1.0 / self.jweight(c)

    def _divmapped(self,u,c):
        return fd.div(u) - self._ell(c) * self._divmiddle(u,c) \
               + (self._ell(c) - 1.0) * self._wzeta(u)

    def _Dmapped(self,u,c):
        return 0.5 * (fd.grad(u)+fd.grad(u).T) - self._ell(c) * self._Mcu(u,c) \
               + (self._ell(c) - 1.0) * self._Lu(u,c)

    def _zcoord(self):  # 3D
        _,_,z = fd.SpatialCoordinate(self.mesh)
        return z

    def _Ftrue(self,u,p,c,v,q,e):
        '''The fully-coupled version including the  x -> xi  mapping and miasma.'''
        divu = self._divmapped(u,c)
        divv = self._divmapped(v,c) # FIXME argue vs. fd.div(v) or _divmapped(v,e)
        Du = self._Dmapped(u,c)
        Dv = self._Dmapped(v,c) # FIXME argue vs. 0.5 * (fd.grad(v)+fd.grad(v).T) or _Dmapped(v,e)
        Du2 = 0.5 * fd.inner(Du, Du) + self.eps * self.Dtyp**2.0
        tau = Bn * Du2**(-1.0/n) * Du  # = 2 nu_e Du
        h = self.hcurrent # FIXME add c(x,y,hcurrent(x,y))?
        cond = fd.conditional(self._zcoord() < h, rho, self.rhom)
        Q0 = fd.FunctionSpace(self.mesh,'DQ',0)
        rhofield = fd.Function(Q0).project(cond)
        source = fd.inner(self._fbody(rhofield),v)
        return (fd.inner(tau, Dv) - p * divv - divu * q - source ) \
                   * self.jweight(c) * fd.dx(degree=self.qdegree) \
               + fd.inner(fd.grad(c),fd.grad(e)) * fd.dx

    def F(self,u,p,c,v,q,e):
        '''Return the nonlinear weak form F(u,p,c;v,q,e) for the coupled
        Glen-Stokes and displacement problems.  Note the weak form for the
        top boundary condition is separate; see Fsmb().'''
        if self.almost:
            return self._Falmost(u,p,c,v,q,e)
        else:
            return self._Ftrue(u,p,c,v,q,e)

    def _tangentu(self,u,z):  # 3D
        return u[0] * z.dx(0) + u[1] * z.dx(1)

    def smbref(self,smb,dt,z):
        '''The surface mass balance value on the top of the reference domain.'''
        return fd.conditional(z > self.Href, dt * smb, dt * smb - self.Href)

    def Fsmb(self,mesh,a,dt,u,c,e):
        '''Return the weak form Fsmb(c;e) of the top boundary condition
        for the displacement problem so we may apply the surface kinematical
        equation weakly.  This weak form also depends on u.'''
        z = self._zcoord()
        smb = a - self._tangentu(u,z) + self._w(u)
        return (c - self.smbref(smb,dt,z)) * e * fd.ds_t


class IceModel2D(IceModel):
    '''The 2D (flowline) case of IceModel.'''

    def _fbody(self, rhofield):
        return fd.as_vector([fd.Constant(0.0), - rhofield * fd.Constant(g)])

    def _czeta(self,c):
        return c.dx(1)

    def _divmiddle(self,u,c):
        return c.dx(0) * u[0].dx(1)

    def _w(self,u):
        return u[1]

    def _wzeta(self,u):
        return u[1].dx(1)

    def _Mcu(self,u,c):
        term12 = 0.5 * c.dx(0) * u[1].dx(1)
        return fd.as_tensor([[c.dx(0) * u[0].dx(1), term12],
                             [term12,               0.0   ]])

    def _Lu(self,u,c):
        return fd.as_tensor([[0.0,              0.5 * u[0].dx(1)],
                             [0.5 * u[0].dx(1), u[1].dx(1)      ]])

    def _zcoord(self):
        _,z = fd.SpatialCoordinate(self.mesh)
        return z

    def _tangentu(self,u,z):
        return u[0] * z.dx(0)

