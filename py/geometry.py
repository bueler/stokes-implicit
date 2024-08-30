# Compute 1D numpy arrays for b(x) and s(x).  Used to set
# initial geometries in the study.

bedtypes = ['flat', 'smooth', 'rough']

# public physics parameters
secpera = 31556926.0    # seconds per year
g, rho = 9.81, 910.0    # m s-2, kg m-3
nglen = 3.0
A3 = 3.1689e-24         # Pa-3 s-1; EISMINT I value of ice softness
B3 = A3**(-1.0/3.0)     # Pa s(1/3);  ice hardness

# private to initial geometry
_len = [120.0e3, 20.0e3, 10.0e3, 4.0e3]  # wavelengths of bed oscillations
_amp = [100.0, 40.0, 20.0, 25.0]         # amplitudes ...
_off = [30.0e3, 0.0, -1.0e3, -400.0]     # offsets ...

# Halfar time-dependent SIA geometry from
#   * P. Halfar (1981), On the dynamics of the ice sheets,
#     J. Geophys. Res. 86 (C11), 11065--11072
# The solution is evaluated at t = t0.  The formula for t0 is
# derived by following equations (3)--(9) in Bueler et al (2005)
# but in 1D.  Only for nglen=3.0.  The formula for t0 comes out
# identical to (9), with f=0 for no isostasy.
_R0 = 65.0e3                             # Halfar dome radius (m)
_H0 = 1400.0                             # Halfar dome height (m)
_alpha = 1.0 / 11.0
_beta = _alpha
_Gamma = 2.0 * A3 * (rho * g)**3.0 / 5.0
t0 = (7.0 / 4.0)**3.0 * (_beta / _Gamma) * _R0**4.0 / _H0**7.0  # public

def _s_halfar(x, t=t0, nglen=3.0):
    import numpy as np
    pp = 1.0 + 1.0 / nglen
    rr = nglen / (2.0 * nglen + 1.0)
    s0 = (t / t0)**_beta * _R0   # margin position at time t
    xsc = (t / t0)**(-_beta) * x[abs(x) < s0] / _R0
    s = np.zeros(np.shape(x))
    s[abs(x) < s0] = _H0 * (t / t0)**(-_alpha) * (1.0 - abs(xsc)**pp)**rr
    s[abs(x) >= s0] = -10000.0  # needs max with bed below
    return s

def halfargeometry(x, t=t0, bed='flat'):
    import numpy as np
    assert bed in bedtypes
    b = np.zeros(np.shape(x))
    if bed != 'flat':
        for k in range(3 if bed == 'smooth' else 4):
            b += _amp[k] * np.sin(2 * np.pi * (x + _off[k]) / _len[k])
    return b, np.maximum(b, _s_halfar(x, t=t))
