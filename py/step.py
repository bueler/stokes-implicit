# TODO re-calculate the weak form to use the geometry determined
#      by b,s in the Stokes parts 3,4 of the weak form

# Compute one-way coupled implicit step of SKE + Stokes problem for
# a 2D glacier, i.e. in (x,z).  The problem has mixed space
#   P1 x DG0^2 x P2^2 x P1
#   (s,  omega,  u,     p)
# where
#   s(x) is surface elevation
#   omega(x) is surface velocity
#   u(x,z) is velocity
#   p(x,z) is pressure
# The weak form consists of four parts:
#   1 SKE for updated s
#   2 weakly-enforce omega=u|_s; omega is surface trace of velocity
#   3 stress balance part of Stokes
#   4 incompressibility part of Stokes
# Run as e.g.:
#   $ python3 onestep.py 100 20

# ISSUES
# 1. This is *not* the full problem because the geometry seen by Stokes
#    does not change; it is the old geometry.
# 2. Seems to run in parallel, but not at e.g. P=6 or higher processes.
#    (Some processes own no ice?)

from sys import argv, path
path.append('../')

# parameters set at runtime
mx = int(argv[1])              # number of elements in x direction
mz = int(argv[2])              # number of elements in z (vertical) direction

from firedrake import *

# TODO move in local versions of the _PinchColumn... objects, unless
# I decide to upgrade StokesExtrude to much-enhanced capability
from stokesextrude import _PinchColumnVelocity, _PinchColumnPressure, printpar
from geometry import secpera, g, rho, nglen, A3, B3, t0, halfargeometry

import numpy as np

# parameters
L = 100.0e3             # domain is (-L,L)
aconst = -2.5e-7        # m s-1
dt = 1.0 * secpera  # dt in years
pvdname = 'result.pvd'

# Stokes parameters
qq = 1.0 / nglen - 1.0
Dtyp = 1.0 / secpera      # = 1 a-1; strain rate scale
eps = 0.0001 * Dtyp**2.0  # viscosity regularization

# set up bm = basemesh once
bm = IntervalMesh(mx, -L, L)
P1bm = FunctionSpace(bm, 'P', 1)
xbm = bm.coordinates.dat.data_ro

# bed, initial geometry as numpy arrays
bed = 'smooth'
printpar(f"Halfar t0 = {t0 / secpera:.3f} a")
b_np, s_initial_np = halfargeometry(xbm, t=t0, bed=bed)

# create the extruded mesh, but leave z coordinate at default
mesh = ExtrudedMesh(bm, mz, layer_height=1.0/mz)

# space for surface elevation and bed elevation *on extruded mesh*
P1R = FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)

# space for surface velocity *on extruded mesh*
P0VR = VectorFunctionSpace(mesh, 'DG', 0, dim=2, vfamily='R', vdegree=0)

# standard P2 x P1 Taylor-Hood spaces for Stokes
P2V = VectorFunctionSpace(mesh, 'P', 2)
P1 = FunctionSpace(mesh, 'P', 1)

# bed elevation, SMB, body force
b = Function(P1R, name='b (m)')
b.dat.data[:] = b_np
a = Constant(aconst)
f_body = Constant((0.0, - rho * g))  # UFL expression only

# old surface elevation must be admissible
sold = Function(P1R, name='s_old (m)')
sold.dat.data[:] = np.maximum(s_initial_np, b_np)

# TODO *don't* do this, except post-solution for viewing
# set extruded mesh geometry from old surface elevation
xorig, zorig = SpatialCoordinate(mesh)
newz = b + (sold - b) * zorig
Vcoord = mesh.coordinates.function_space()
newcoord = Function(Vcoord).interpolate(as_vector([xorig, newz]))
mesh.coordinates.assign(newcoord)

# mixed space: (surface elevation, surface velocity, velocity, pressure)
Z = P1R * P0VR * P2V * P1    # product space
soup = Function(Z)
printpar(f'dimensions: n_s = {P1R.dim()}, n_omega = {P0VR.dim()}, n_u = {P2V.dim()}, n_p = {P1.dim()}')

# initialize s; note sold is admissible
soup.subfunctions[0].interpolate(sold)

# TODO use following to build new thickness-transformed weak form
# for reference: these seem to be how "grad()" and "div()" are implemented
#   by UFL, because replacing in the weak form F below does not change values
#def mygrad(u):
#    return as_matrix([[u[0].dx(0), u[0].dx(1)], [u[1].dx(0), u[1].dx(1)]])
#def mydiv(u):
#    return u[0].dx(0) + u[1].dx(1)

# helper UFL expressions for weak form
def _D(u):
    # strain rate tensor
    return 0.5 * (grad(u) + grad(u).T)
def _n(s):
    # upward surface normal vector
    return as_vector([-s.dx(0), Constant(1.0)])
def _S(s):
    # inverse of surface area density
    # note for base mesh:  dx_bm = _S(s) ds_t
    return (1.0 + s.dx(0) * s.dx(0))**(-0.5)

# weak form for the coupled (SKE+Stokes) problem
s, omega, u, p = split(soup)
r, zeta, v, q = TestFunctions(Z)
Du2 = 0.5 * inner(_D(u), _D(u)) + eps
ell = sold + dt * a
F = inner(s - dt * dot(omega, _n(s)) - ell, r) * _S(s) * ds_t(degree=2)  # term 1 with r
F += inner(omega - u, zeta) * ds_t                                       # term 2 with zeta
F += inner(B3 * Du2**(qq / 2.0) * _D(u), _D(v)) * dx(degree=3)           # term 3 with v
F -= (p * div(v) + inner(f_body, v)) * dx
F -= div(u) * q * dx                                                     # term 4 with q

# conditions; the pinch conditions should be updated in coupled problem
conds = [ DirichletBC(Z.sub(0), b, (1,2)),                         # s=b at ends
          DirichletBC(Z.sub(2), Constant((0.0, 0.0)), (1,2)),      # u=0 at ends
          DirichletBC(Z.sub(2), Constant((0.0, 0.0)), 'bottom') ]  # u=0 at base (no sliding)
pinchU = _PinchColumnVelocity(Z.sub(2), b, sold, htol=1.0, dim=2)
pinchP = _PinchColumnPressure(Z.sub(3), b, sold, htol=1.0)
conds += [ pinchU, pinchP ]

# bounds for VI solver
boundINF  = 1.0e100   # versus PETSc.INFINITY = 4.5e307 which causes overflow inside numpy
infU  = as_vector([boundINF, boundINF])
soupl = Function(Z)
soupl.subfunctions[0].interpolate(b)          # the nontrivial one!
soupl.subfunctions[1].interpolate(-infU)
soupl.subfunctions[2].interpolate(-infU)
soupl.subfunctions[3].interpolate(-boundINF)
soupu = Function(Z)
soupu.subfunctions[0].interpolate(boundINF)
soupu.subfunctions[1].interpolate(infU)
soupu.subfunctions[2].interpolate(infU)
soupu.subfunctions[3].interpolate(boundINF)
mybounds = (soupl, soupu)

# solver parameters
par = { "snes_type": "vinewtonrsls",
        "snes_linesearch_type": "basic",
        #"snes_linesearch_type": "bt",  # better?
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_vi_zero_tolerance": 1.0e-8,
        "snes_max_it": 200,
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-12,
        "snes_stol": 0.0,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_shift_type": "inblocks",
        "pc_factor_mat_solver_type": "mumps" }

# view mat in Matlab format; slow at higher resolution or in parallel
if False:
    par.update({"ksp_view_mat": ":foo.m:ascii_matlab"})
    # to view:
    # >> foo
    # >> A = Mat_0x84000003_K;  # last index K for converged state
    # >> spy(A)

# set up solver
problem = NonlinearVariationalProblem(F, soup, bcs=conds)
solver = NonlinearVariationalSolver(problem,
                                    options_prefix='coupled',
                                    solver_parameters=par)

# solve
solver.solve(bounds=mybounds)

# diagnostic: surface motion from surface elevation and velocity solution
def _surface_motion(s, omega):
    P0R = FunctionSpace(mesh, 'DG', 0, vfamily='R', vdegree=0)
    Phi = Function(P0R).interpolate(- dot(omega, _n(s)))
    Phi.rename(f'Phi = -u|_s . n_s (m s-1)')
    return Phi

# diagnostic: effective viscosity nu from the velocity solution
def _effective_viscosity(u):
    Du2 = 0.5 * inner(_D(u), _D(u))
    nueps = Function(P1).interpolate(0.5 * B3 * (Du2  + (eps * Dtyp)**2)**(qq/2.0))
    nueps.rename(f'nu (eps={eps:.3f}; Pa s)')
    return nueps

# diagnostic: hydrostatic pressure from geometry
def _p_hydrostatic(z, s):
    phydro = Function(P1).interpolate(rho * g * (s - z))
    phydro.rename('p_hydro (Pa)')
    return phydro

# get geometry fields and diagnostics
s, omega = soup.subfunctions[0], soup.subfunctions[1] 
s.rename('surface elevation (R space; m)')
omega.rename('surface velocity (R space; m s-1)')
sdiff = Function(P1).interpolate(s - sold)
sdiff.rename('sdiff = s - sold (m)')
Phi = _surface_motion(s, omega)

# get dynamical fields and diagnostics
u, p = soup.subfunctions[2], soup.subfunctions[3] 
u.rename('velocity (m s-1)')
p.rename('pressure (Pa)')
nueps = _effective_viscosity(u)
_, z = SpatialCoordinate(mesh)
phydro = _p_hydrostatic(z, s)
pdiff = Function(P1).interpolate(p - phydro)
pdiff.rename('pdiff = phydro - p (Pa)')

# save results
printpar(f'opening {pvdname} ...')
outfile = VTKFile(pvdname)
outfile.write(b, s, sold, sdiff, omega, u, p, Phi, nueps, pdiff)
