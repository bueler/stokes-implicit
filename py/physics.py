# Almost dimension-independent code to set up the Stokes problem and
# associated constructions.

import firedrake as fd
from stokesextrude import extend_p1_from_basemesh

# public parameters
secpera = 31556926.0  # seconds per year
g, rho = 9.81, 910.0  # m s-2, kg m-3
nglen = 3.0
A3 = 3.1689e-24  # Pa-3 s-1; EISMINT I value of ice softness
B3 = A3 ** (-1.0 / 3.0)  # Pa s(1/3);  ice hardness


def _D(w):
    return 0.5 * (fd.grad(w) + fd.grad(w).T)


# weak form for the Stokes problem; se is a StokesExtrude object
def form_stokes(
    se,
    sR,
    mu0=0.0,
    fssa=True,
    dt_fssa=0.0,
    smb_fssa=None,
):
    u, p = fd.split(se.up)
    v, q = fd.TestFunctions(se.Z)
    Du2 = 0.5 * fd.inner(_D(u), _D(u)) + mu0
    pp = (1.0 / nglen) + 1.0
    qqq = (pp - 2.0) / 2.0
    F = fd.inner(B3 * Du2 ** qqq * _D(u), _D(v)) * fd.dx(
        degree=4
    )  # correspondence with paper: nu_p = 0.5 B3
    F -= (p * fd.div(v) + fd.div(u) * q) * fd.dx
    source = fd.inner(se.f_body, v) * fd.dx
    if fssa:
        # see section 4.2 in Lofgren et al (2022)
        assert se.dim in [2, 3]
        if se.dim == 2:
            nsR = fd.as_vector([-sR.dx(0), fd.Constant(1.0)])
            n = nsR / fd.sqrt(sR.dx(0) ** 2 + 1.0)
            z = fd.Constant(fd.as_vector([0.0, 1.0]))
        else:
            nsR = fd.as_vector([-sR.dx(0), -sR.dx(1), fd.Constant(1.0)])
            n = nsR / fd.sqrt(sR.dx(0) ** 2 + sR.dx(1) ** 2 + 1.0)
            z = fd.Constant(fd.as_vector([0.0, 0.0, 1.0]))
        theta_fssa = 1.0
        C = theta_fssa * dt_fssa
        F -= C * fd.inner(u, n) * fd.inner(se.f_body, v) * fd.ds_t
        aR = extend_p1_from_basemesh(se.mesh, smb_fssa)
        source += C * aR * fd.inner(z, n) * fd.inner(se.f_body, v) * fd.ds_t
    F -= source
    return F


# diagnostic: effective viscosity nu from the velocity solution
def effective_viscosity(u, P1, mu0=0.0):
    Du2 = 0.5 * fd.inner(_D(u), _D(u))
    pp = (1.0 / nglen) + 1.0
    qqq = (pp - 2.0) / 2.0
    nu = fd.Function(P1).interpolate(0.5 * B3 * Du2 ** qqq)  # vs paper: nu_p = 0.5 B3
    nu.rename("nu (unregularized; Pa s)")
    Du2 += mu0
    nueps = fd.Function(P1).interpolate(0.5 * B3 * Du2 ** qqq)
    nueps.rename(f"nu (mu0={mu0:.3f}; Pa s)")
    return nu, nueps


# diagnostic: hydrostatic pressure
def p_hydrostatic(se, sR, P1):
    x = fd.SpatialCoordinate(se.mesh)
    phydro = fd.Function(P1).interpolate(rho * g * (sR - x[se.dim - 1]))
    phydro.rename("p_hydro (Pa)")
    return phydro


# Compute the surface motion operator Phi = - <u,w>|_s . n_s.
# This function returns a UFL expression, so q can be a TestFunction
# or a Function.
def Phi(dim, s, us, q, H0=1000.0):
    assert dim in [2, 3]
    if dim == 2:
        ns = fd.as_vector([-s.dx(0), fd.Constant(1.0)])
    else:
        ns = fd.as_vector([-s.dx(0), -s.dx(1), fd.Constant(1.0)])
    Phi = -fd.dot(us, ns) * q
    return Phi
