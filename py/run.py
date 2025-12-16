from sys import argv
from firedrake import *

from physics import secpera, g, rho, form_stokes, effective_viscosity, Phi
from stokesextrude import StokesExtrude, SolverParams, trace_vector_to_p2, printpar
from geometryinit import generategeometry

# FIXME consider this later: from viamr import VIAMR

# parameters set at runtime
assert len(argv) >= 6  # 0 is "run.py", plus next five
bdim = int(argv[1])  # dimension of base (map-plane) mesh
mx = int(argv[2])  # number of elements in x (*and* y if bdim > 1) directions
mz = int(argv[3])  # number of elements in z (vertical) direction
Nsteps = int(argv[4])  # number of time steps
dt = float(argv[5]) * secpera  # dt in years, converted to seconds

# method string is length 3:
#   000  explicit scheme with no extrapolations, stabilizations, or semi-implicitness
#   F    classic FSSA extrapolation in body force (Lofgren 2022)
#   S    symmetrized FSSA (Tominec et al 2025)  FIXME not implemented yet
#    E   implicit edge stabilization as written (Tominec et al 2025)
#    H   implicit edge stabilization but with horizontal velocity in coefficient
#     S  implicit surface in surface-motion term ("semi-implicit" since velocity is old)
if len(argv) > 6:
    method = argv[6]  # it is a string
    assert len(method) == 3
    assert method[0] in ["0", "F", "S"]
    assert method[1] in ["0", "E", "H"]
    assert method[2] in ["0", "S"]
else:
    method = "FES"  # current default

show_fig_2D = True  # if True, show final geometry solution (serial only)
save_true_stokes = False  # if True, save Stokes solution for final geometry

# experiment parameters
L = 100.0e3  # map-plane domain is (-L,L) or (-L,L)x(-L,L)

# Stokes parameters
Dtyp = 1.0 / secpera  # = 1 a-1; strain rate scale
mu0 = 0.0001 * Dtyp ** 2.0  # viscosity regularization

# set up bm = basemesh once
assert bdim in [1, 2]
if bdim == 1:
    bm = IntervalMesh(mx, -L, L)
else:
    bm = RectangleMesh(mx, mx, L, L, originX=-L, originY=-L)

# bed and initial geometry
P1bm = FunctionSpace(bm, "CG", 1)
xbm = SpatialCoordinate(bm)
b, s_initial = generategeometry(P1bm, xbm, bdim=bdim)
s_initial.rename("s^{0}")

# surface mass balance (SMB) function; initialized to zero
a = Function(P1bm, name="surface mass balance (m s-1)")

# create the extruded mesh, but leave z coordinate at default
se = StokesExtrude(bm, mz=mz, htol=1.0)
x = SpatialCoordinate(se.mesh)

# set up Stokes problem: elements and boundary conditions
se.mixed_TaylorHood()
if bdim == 1:
    se.body_force(Constant((0.0, -rho * g)))
    se.dirichlet((1, 2), Constant((0.0, 0.0)))  # lateral velocity
    se.dirichlet(("bottom",), Constant((0.0, 0.0)))
else:
    se.body_force(Constant((0.0, 0.0, -rho * g)))
    se.dirichlet((1, 2, 3, 4), Constant((0.0, 0.0, 0.0)))  # lateral velocity
    se.dirichlet(("bottom",), Constant((0.0, 0.0, 0.0)))

# set up Stokes solver
params = SolverParams["newton"]
params.update(SolverParams["mumps"])
# params.update(SolverParams["schur_nonscalable_pinch"])
params.update({"snes_monitor": None})
params.update({"snes_converged_reason": None})
params.update({"ksp_converged_reason": None})
params.update({"snes_atol": 1.0e-2})
params.update({"snes_linesearch_type": "bt"})  # helps with non-flat beds, it seems
# params.update({"snes_view": None})

# weak form for semi-implicit method, documented in paper, using old velocity in weak form
if bdim == 1:
    sibcs = [DirichletBC(P1bm, b, (1, 2))]  # s = b at lateral boundary
else:
    sibcs = [DirichletBC(P1bm, b, (1, 2, 3, 4))]  # s = b at lateral boundary
siq = TestFunction(P1bm)
sisold = Function(P1bm, name="s_old")
siP2Vbm = VectorFunctionSpace(bm, "CG", 2, dim=bdim + 1)
siubm = Function(siP2Vbm)  # surface velocity
s = Function(P1bm, name="s")  # solution surface
siF = inner(s - (sisold + dt * a), siq) * dx

# explicit or semi-implicit for surface motion
if method[2] == "0":
    siF += dt * Phi(se.dim, sisold, siubm, siq) * dx
elif method[2] == "S":
    siF += dt * Phi(se.dim, s, siubm, siq) * dx

# edge stabilization
n = FacetNormal(bm)
h = CellSize(bm)
if method[1] == "0":
    # no edge stabilization
    pass
elif method[1] == "E":
    # equation (4.1) in Tominec et al (2025) manuscript
    gamma = 0.5 * avg(h ** 2 * sqrt(dot(siubm, siubm)))
    siF += dt * gamma * jump(grad(s), n) * jump(grad(siq), n) * dS
elif method[1] == "H":
    # equation (4.1) in Tominec et al (2025) manuscript, but with
    #   horizontal velocity in scaling
    if bdim == 1:
        magHOR = sqrt(siubm[0] * siubm[0])
    else:
        uHOR = as_vector([siubm[0], siubm[1]])
        magHOR = sqrt(dot(uHOR, uHOR))
    gamma = 0.5 * avg(h ** 2 * magHOR)
    siF += dt * gamma * jump(grad(s), n) * jump(grad(siq), n) * dS

# solver for semi-implicit method
siparams = {
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_rtol": 1.0e-6,
    "snes_atol": 1.0e-6,
    "snes_stol": 0.0,
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-8,
    "snes_linesearch_type": "basic",
    "snes_max_it": 200,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
siproblem = NonlinearVariationalProblem(siF, s, sibcs)
sisolver = NonlinearVariationalSolver(
    siproblem, solver_parameters=siparams, options_prefix="step"
)
siINF = Function(P1bm).interpolate(Constant(PETSc.INFINITY))

# start the solution process
meshstr = f"{mx} x {mx} x {mz}" if se.dim == 3 else f"{mx} x {mz}"
printpar(f"doing {Nsteps} steps of dt = {dt/secpera:.3f} a")
printpar(f"  solving Stokes + SKE on {meshstr} extruded {se.dim}D mesh ...")
printpar(f"  dimensions: n_u = {se.V.dim()}, n_p = {se.W.dim()}, n_s = {P1bm.dim()}")

# time-stepping loop
t = 0.0
s.interpolate(s_initial)
for n in range(Nsteps):
    # set geometry (z coordinate) of extruded mesh
    se.reset_elevations(b, s)
    P1R = FunctionSpace(se.mesh, "P", 1, vfamily="R", vdegree=0)
    sR = Function(P1R)
    sR.dat.data_with_halos[:] = s.dat.data_ro_with_halos

    # solve Stokes on extruded mesh with some type of extrapolation in the body force
    if method[0] == "0":
        stokesF = form_stokes(se, sR, mu0=mu0, fssa=False)
    elif method[0] == "F":
        stokesF = form_stokes(se, sR, mu0=mu0, fssa=True, dt_fssa=dt, smb_fssa=a)
    elif method[0] == "S":
        raise NotImplementedError
    u, p = se.solve(F=stokesF, par=params, zeroheight="indices")

    # extract surface velocity (trace) in m s-1
    ubm = trace_vector_to_p2(bm, se.mesh, u, dim=se.dim)

    # time step of free-boundary problem for surface elevation
    #   using surface velocity from old surface elevation
    sisold.dat.data[:] = s.dat.data_ro
    siubm.dat.data[:] = ubm.dat.data_ro
    sisolver.solve(bounds=(b, siINF))

    # update time
    t += dt
    printpar(f"t = {t/secpera:.3f} a done")

# at final time, write map-plane fields
namemap = f"result_map.pvd"
filemap = VTKFile(namemap)
dstotal = Function(P1bm, name="ds_total = s - s^{0}").interpolate(s - s_initial)
dslast = Function(P1bm, name="ds_last = s - s^{n-1}").interpolate(s - sisold)
printpar(f"  writing {bdim}D fields to {namemap} ...")
if bm.comm.size > 1:
    rankbm = Function(FunctionSpace(bm, "DG", 0))
    rankbm.dat.data[:] = bm.comm.rank
    rankbm.rename("rank")
    filemap.write(s, b, s_initial, dstotal, dslast, rankbm, time=t)
else:
    filemap.write(s, b, s_initial, dstotal, dslast, time=t)

# optionally show a figure if appropriate (better than paraview!)
if show_fig_2D and bdim == 1 and bm.comm.size == 1:
    printpar(f"  generating Matplotlib figure with {bdim}D fields ...")
    import matplotlib.pyplot as plt
    xx = bm.coordinates.dat.data_ro
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(xx / 1.0e3, s.dat.data_ro, color="C1", label=r"$s$")
    ax1.plot(xx / 1.0e3, s_initial.dat.data_ro, "--", color="C1", label=r"$s^{0}$")
    ax1.plot(xx / 1.0e3, b.dat.data_ro, color="k", label=r"$b$")
    ax1.legend(loc="upper left")
    ax1.set_xticklabels([])
    ax1.grid(visible=True)
    ax1.set_ylabel("elevation (m)")
    ax2.plot(xx / 1.0e3, dstotal.dat.data_ro, ".", color="C1", label=r"$s - s^{0}$")
    ax2.plot(xx / 1.0e3, dslast.dat.data_ro, ".", color="C2", label=r"$s - s^{n-1}$")
    ax2.legend(loc="upper right")
    ax2.set_ylabel(r"m")
    ax2.grid(visible=True)
    plt.xlabel("x (km)")
    plt.show()

# if desired, solve Stokes over final geometry, and write 3D fields into .pvd
if save_true_stokes:
    printpar(f"  solving Stokes equations for final geometry (diagnostic) ...")
    stokesF = form_stokes(se, sR, mu0=mu0, fssa=False)
    u, p = se.solve(F=stokesF, par=params, zeroheight="indices")
    ubm = trace_vector_to_p2(bm, se.mesh, u, dim=se.dim)
    namestokes = f"result_stokes.pvd"
    filestokes = VTKFile(namestokes)
    printpar(f"  writing {se.dim}D fields to {namestokes} ...")
    u.rename("velocity (m s-1)")
    p.rename("pressure (Pa)")
    P1 = FunctionSpace(se.mesh, "CG", 1)
    nu, nueps = effective_viscosity(u, P1, mu0=mu0)
    filestokes.write(u, p, nu, nueps, time=t)
