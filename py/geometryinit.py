# Compute initial s(x) from Halfar formulas, and b(x) from ad hoc sum of modes.

import firedrake as fd
from physics import A3, rho, g

R0 = 65.0e3                             # Halfar dome radius (m)
H0 = 1400.0                             # Halfar dome height (m)

def _t0_halfar(bdim=2):
    assert bdim in [1, 2]
    if bdim == 1:
        _alpha = 1.0 / 11.0
        _beta = _alpha
    else:
        _alpha = 1.0 / 9.0
        _beta = 1.0 / 18.0
    _Gamma = 2.0 * A3 * (rho * g)**3.0 / 5.0
    t0 = (7.0 / 4.0)**3.0 * (_beta / _Gamma) * R0**4.0 / H0**7.0
    return t0

# 2D Halfar time-dependent SIA geometry from
#   P. Halfar (1983). On the dynamics of the ice sheets 2.
#   J. Geophys. Res., 88 (C10), 6043--6051
# The solution is evaluated at t = t0.  The formula for t0 is
# derived from equations (3)--(9) in
#   Bueler and others (2005). Exact solutions and verification of numerical
#   models for isothermal ice sheets. J. Glaciol. 51 (173), 291--306
# Only for nglen=3.0.  See formula (9) for t0 in 2D, with f=0 for no isostasy.
# The returned s is a UFL expression.  It is invalid outside of the ice, and
# it generally needs a max() operation with a bed map; see generategeometry().
def _s_2d_halfar(V, x, y, t=None):
    _alpha = 1.0 / 9.0
    _beta = 1.0 / 18.0
    t0 = _t0_halfar(2)
    if t is None:
        t = t0
    pp = 1.0 + 1.0 / 3.0
    rr = 3.0 / (2.0 * 3.0 + 1.0)
    s0 = (t / t0)**_beta * R0   # margin position at time t
    rsc = (t / t0)**(-_beta) * fd.sqrt(x * x + y * y) / R0
    s = fd.conditional(x * x + y * y < s0 * s0,
                       H0 * (t / t0)**(-_alpha) * (1.0 - rsc**pp)**rr,
                       fd.Constant(-10000.0))
    return s

# 1D Halfar time-dependent SIA geometry from
#   P. Halfar (1981). On the dynamics of the ice sheets,
#   J. Geophys. Res. 86 (C11), 11065--11072
# Same as _s_2d_halfar(), but note that the formula for t0 comes out
# identical to (9).
def _s_1d_halfar(x, t=None):
    _alpha = 1.0 / 11.0
    _beta = _alpha
    t0 = _t0_halfar(1)
    if t is None:
        t = t0
    pp = 1.0 + 1.0 / 3.0
    rr = 3.0 / (2.0 * 3.0 + 1.0)
    s0 = (t / t0)**_beta * R0   # margin position at time t
    xsc = (t / t0)**(-_beta) * x / R0
    s = fd.conditional(fd.abs(x) < s0,
                       H0 * (t / t0)**(-_alpha) * (1.0 - fd.abs(xsc)**pp)**rr,
                       fd.Constant(-10000.0))
    return s

# private to bed geometry
_len = [120.0e3, 20.0e3, 10.0e3, 4.0e3]  # wavelengths of bed oscillations
_amp = [100.0, 40.0, 20.0, 25.0]         # amplitudes ...
_off = [30.0e3, 0.0, -1.0e3, -400.0]     # offsets ...

# generate rough bed geometry in x or in (x,y)
# assumes input x is from "x = SpatialCoordinate(mesh)", so x=x[0] and y=x[1] in 2D
# FIXME in 2D this comes out rough only in x variable
def generategeometry(V, x, t=None, bdim=2):
    b_ufl = 0.0
    if bdim == 1:
        for k in range(4):
            off = fd.Constant(_off[k])
            b_ufl += _amp[k] * fd.sin(2 * fd.pi * (x + off) / _len[k])
        sh = _s_1d_halfar(V, x, t=t)
    else:
        for k in range(4):
            #off = fd.Constant(_off[k])
            b_ufl += _amp[k] * fd.sin(2 * fd.pi * (x[0] + _off[k]) / _len[k])
        sh = _s_2d_halfar(V, x[0], x[1], t=t)
    b = fd.Function(V, name="bed elevation").interpolate(b_ufl)
    s_ufl = fd.conditional(sh > b, sh, b)
    s = fd.Function(V, name="surface elevation").interpolate(s_ufl)
    return b, s
