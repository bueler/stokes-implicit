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


# weak form for the Stokes problem
def form_stokes(
    se,  # StokesExtrude object
    sR,  # surface elevation s in the R space
    mu0=0.0,  # Glen law regularization
    qdegree=4,  # quadrature degree
    stab=None,  # stabilization type: None, "fssa", "sym"
    dt_stab=0.0,
    smb_stab=None,
):
    u, p = fd.split(se.up)
    v, q = fd.TestFunctions(se.Z)
    Du2 = 0.5 * fd.inner(se.D(u), se.D(u)) + mu0
    pp = (1.0 / nglen) + 1.0
    qqq = (pp - 2.0) / 2.0
    # correspondence with paper: nu_p = 0.5 B3
    F = B3 * Du2 ** qqq * fd.inner(se.D(u), se.D(v)) * fd.dx(degree=qdegree)
    F -= (p * fd.div(v) + fd.div(u) * q) * fd.dx
    assert se.dim in [2, 3]
    if se.dim == 2:
        z = fd.Constant(fd.as_vector([0.0, 1.0]))
    else:
        z = fd.Constant(fd.as_vector([0.0, 0.0, 1.0]))
    source = -rho * g * fd.inner(z, v) * fd.dx
    if stab is not None:
        assert stab in ["fssa", "sym"]
        if se.dim == 2:
            nsR = fd.as_vector([-sR.dx(0), fd.Constant(1.0)])
            omega = fd.sqrt(sR.dx(0) ** 2 + 1.0)
        else:
            nsR = fd.as_vector([-sR.dx(0), -sR.dx(1), fd.Constant(1.0)])
            omega = fd.sqrt(sR.dx(0) ** 2 + sR.dx(1) ** 2 + 1.0)
        n = nsR / omega
        aR = extend_p1_from_basemesh(se.mesh, smb_stab)
        if stab == "fssa":
            # see section 4.2 in Lofgren et al (2022); using theta = 1.0
            F += rho * g * dt_stab * fd.inner(u, n) * fd.inner(v, z) * fd.ds_t
            source -= rho * g * dt_stab * aR * fd.inner(z, n) * fd.inner(v, z) * fd.ds_t  # FIXME correct?
        elif stab == "sym":
            # see section (4.26) in Tominec et al (2025)
            F += 0.5 * rho * g * dt_stab * omega * fd.inner(u, n) * fd.inner(v, n) * fd.ds_t
            source -= rho * g * dt_stab * aR * fd.inner(v, n) * fd.ds_t  # FIXME correct?
    F -= source
    return F


# diagnostic: effective viscosity nu from the velocity solution
def effective_viscosity(se, u, P1, mu0=0.0):
    Du2 = 0.5 * fd.inner(se.D(u), se.D(u))
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
