# module for defining UFL functionals for coupled weak form

import firedrake as fd
from .constants import g,rho,n,Bn  # for default values

__all__ = ['IceModel', 'IceModel2D']

class IceModel(object):
    '''Physics of the coupled ice flow and surface kinematical problems.'''

    def __init__(self, mesh=None, Href=None, eps=None, Dtyp=None):
        self.mesh = mesh
        self.Href = Href
        self.eps = eps
        self.Dtyp = Dtyp
        self.delta = 0.1
        self.qdegree = 3  # used in mapped weak form FIXME how to determine a wise value?
        self.k = self.mesh._base_mesh.cell_dimension()

    def _fbody(self):  # 3D
        return fd.as_vector([fd.Constant(0.0), fd.Constant(0.0), - rho * fd.Constant(g)])

    def _divmiddle(self,u,c):  # 3D
        return c.dx(0) * u[0].dx(2) + c.dx(1) * u[1].dx(2)

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

    def _tangentu(self,u,z):  # 3D
        return u[0] * z.dx(0) + u[1] * z.dx(1)

    def jweight(self,c):
        xx = 1.0 + c.dx(self.k)
        # alternative:
        #return fd.max_value(xx, self.delta)
        return 0.5 * (xx + fd.sqrt(xx**2 + self.delta**2))

    def _ell(self,c):
        return 1.0 / self.jweight(c)

    def _divmapped(self,u,c):
        wzeta = u[self.k].dx(self.k)
        return fd.div(u) - self._ell(c) * self._divmiddle(u,c) \
               + (self._ell(c) - 1.0) * wzeta

    def _Dmapped(self,u,c):
        return 0.5 * (fd.grad(u)+fd.grad(u).T) - self._ell(c) * self._Mcu(u,c) \
               + (self._ell(c) - 1.0) * self._Lu(u,c)

    def F(self,u,p,c,v,q,e):
        '''Return the nonlinear weak form F(u,p,c;v,q,e) for the coupled
        Glen-Stokes and displacement problems, based on the  x -> xi  mapping.
        Note the top boundary condition weak form is separate; see Fsmb().'''
        divu = self._divmapped(u,c)
        divv = self._divmapped(v,c)
        Du = self._Dmapped(u,c)
        Dv = self._Dmapped(v,c)
        Du2 = 0.5 * fd.inner(Du, Du) + self.eps * self.Dtyp**2.0
        tau = Bn * Du2**(-1.0/n) * Du  # = 2 nu_e Du
        source = fd.inner(self._fbody(),v)
        return (fd.inner(tau, Dv) - p * divv - divu * q - source) \
                   * self.jweight(c) * fd.dx(degree=self.qdegree) \
               + fd.inner(fd.grad(c),fd.grad(e)) * fd.dx

    def _psi(self,a,dt,u,c):
        '''The top-surface boundary value for c.  Where ice is present this
        is built using the surface mass balance value.'''
        # FIXME only set up for b=0
        # FIXME criteria for significant ice is lame: "h > 1.1 Href"
        # FIXME assumes h^{n-1} = lambda (where there is significant ice)
        x = fd.SpatialCoordinate(self.mesh)
        h = x[self.k] + c;
        dhdt = a - self._tangentu(u,x[self.k]) + u[self.k]
        return fd.conditional(h > 1.1*self.Href, dt * dhdt, 0.0-x[self.k])

    def B(self,a,dt,u,c,e):
        '''Return the weak form of the top boundary condition
        for the displacement problem: we apply the surface kinematical
        equation weakly.'''
        return (c - self._psi(a,dt,u,c)) * e * fd.ds_t


class IceModel2D(IceModel):
    '''The 2D (flowline) case of IceModel.'''

    def _fbody(self):
        return fd.as_vector([fd.Constant(0.0), - rho * fd.Constant(g)])

    def _divmiddle(self,u,c):
        return c.dx(0) * u[0].dx(1)

    def _Mcu(self,u,c):
        term12 = 0.5 * c.dx(0) * u[1].dx(1)
        return fd.as_tensor([[c.dx(0) * u[0].dx(1), term12],
                             [term12,               0.0   ]])

    def _Lu(self,u,c):
        return fd.as_tensor([[0.0,              0.5 * u[0].dx(1)],
                             [0.5 * u[0].dx(1), u[1].dx(1)      ]])

    def _tangentu(self,u,z):
        return u[0] * z.dx(0)

