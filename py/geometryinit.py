# Compute initial s(x) from Halfar formulas, and b(x) from ad hoc sum of modes.

import numpy as np
from physics import A3, rho, g

_R0 = 65.0e3                             # Halfar dome radius (m)
_H0 = 1400.0                             # Halfar dome height (m)
_Gamma = 2.0 * A3 * (rho * g)**3.0 / 5.0

t0 = -1.0   # public and reset by calls below ... embarrassing design

# 2D Halfar time-dependent SIA geometry from
#   P. Halfar (1983). On the dynamics of the ice sheets 2.
#   J. Geophys. Res., 88 (C10), 6043--6051
# The solution is evaluated at t = t0.  The formula for t0 is
# derived from equations (3)--(9) in
#   Bueler and others (2005). Exact solutions and verification of numerical
#   models for isothermal ice sheets. J. Glaciol. 51 (173), 291--306
# Only for nglen=3.0.  The formula for t0 comes out
# identical to (9), with f=0 for no isostasy.

def _s_2d_halfar(x, y, t=None, nglen=3.0):
    _alpha = 1.0 / 9.0
    _beta = 1.0 / 18.0
    t0 = (7.0 / 4.0)**3.0 * (_beta / _Gamma) * _R0**4.0 / _H0**7.0
    if t is None:
        t = t0
    pp = 1.0 + 1.0 / nglen
    rr = nglen / (2.0 * nglen + 1.0)
    s0 = (t / t0)**_beta * _R0   # margin position at time t
    ice = (x * x + y * y < s0 * s0)
    rsc = (t / t0)**(-_beta) * np.sqrt(x[ice] * x[ice] + y[ice] * y[ice]) / _R0
    s = np.zeros(np.shape(x))
    s[:] = -10000.0  # needs max with bed below
    s[ice] = _H0 * (t / t0)**(-_alpha) * (1.0 - rsc**pp)**rr
    return s, t0

# 1D Halfar time-dependent SIA geometry from
#   P. Halfar (1981). On the dynamics of the ice sheets,
#   J. Geophys. Res. 86 (C11), 11065--11072
# The solution is evaluated at t = t0.  The formula for t0 is
# derived by following equations (3)--(9) in Bueler et al (2005)
# but in 1D.  Only for nglen=3.0.  The formula for t0 comes out
# identical to (9), with f=0 for no isostasy.
def _s_1d_halfar(x, t=None, nglen=3.0):
    _alpha = 1.0 / 11.0
    _beta = _alpha
    _Gamma = 2.0 * A3 * (rho * g)**3.0 / 5.0
    t0 = (7.0 / 4.0)**3.0 * (_beta / _Gamma) * _R0**4.0 / _H0**7.0
    if t is None:
        t = t0
    pp = 1.0 + 1.0 / nglen
    rr = nglen / (2.0 * nglen + 1.0)
    s0 = (t / t0)**_beta * _R0   # margin position at time t
    xsc = (t / t0)**(-_beta) * x[abs(x) < s0] / _R0
    s = np.zeros(np.shape(x))
    s[:] = -10000.0  # needs max with bed below
    s[abs(x) < s0] = _H0 * (t / t0)**(-_alpha) * (1.0 - abs(xsc)**pp)**rr
    return s

# private to bed geometry
_len = [120.0e3, 20.0e3, 10.0e3, 4.0e3]  # wavelengths of bed oscillations
_amp = [100.0, 40.0, 20.0, 25.0]         # amplitudes ...
_off = [30.0e3, 0.0, -1.0e3, -400.0]     # offsets ...

def generategeometry(x, t=t0, bed='flat'):
    import numpy as np
    assert bed in bedtypes
    b = np.zeros(np.shape(x))
    if bed != 'flat':
        for k in range(3 if bed == 'smooth' else 4):
            b += _amp[k] * np.sin(2 * np.pi * (x + _off[k]) / _len[k])
    return b, np.maximum(b, _s_halfar(x, t=t))
