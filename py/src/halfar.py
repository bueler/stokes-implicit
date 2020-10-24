# module for Halfar solutions: 2D = Halfar, 1981; 3D = Halfar, 1983

import firedrake as fd
from .constants import secpera,g,rho,n,An,Bn,Gamma

__all__ = ['halfar2d', 'halfar3d']

inpow = (n+1) / n
outpow = n / (2*n+1)

assert n == 3.0   # only written for this case

# see equation (9) in Bueler et al (2005)
def _gett0(beta,R0,H0):
    t0 = (beta/Gamma) * (7.0/4.0)**3.0 * (R0**4.0 / H0**7.0)
    return t0

# Halfar (1981) solution
def halfar2d(x,R0=500.0e3,H0=3000.0):
    alpha = 1.0/11.0
    beta = alpha
    absx = fd.sqrt(x*x)
    inside = fd.max_value(0.0, 1.0 - pow(absx/R0,inpow))
    return _gett0(beta,R0,H0), H0 * pow(inside,outpow)

# Halfar (1983) solution
def halfar3d(x,y,R0=500.0e3,H0=3000.0):
    alpha = 1.0/9.0
    beta = 1.0/18.0
    r = fd.sqrt(x*x + y*y)
    inside = fd.max_value(0.0, 1.0 - pow(r/R0,inpow))
    return _gett0(beta,R0,H0), H0 * pow(inside,outpow)
