from sys import argv
import numpy as np
from firedrake import *

from stokesextrude import StokesExtrude, SolverParams, trace_vector_to_p2, printpar
from physics import secpera, g, rho, nglen, form_stokes, effective_viscosity, p_hydrostatic, Phi
from geometryinit import bedtypes, t0, generategeometry
from figures import mkdir, livefigure, snapsfigure
from measure import geometryreport, sampleratios

# parameters set at runtime
mx = int(argv[1])              # number of elements in x direction
mz = int(argv[2])              # number of elements in z (vertical) direction
Nsteps = int(argv[3])          # number of time steps
dt = float(argv[4]) * secpera  # dt in years, converted to seconds
bed = argv[5]                  # 'flat', 'smooth', 'rough'
ratiosfile = argv[6]           # at the end, append a pair of ratios into this file
writepng = (len(argv) > 7)
if writepng:
    dirroot = argv[7]
writepvd = (len(argv) > 8)
if writepvd:
    pvdroot = argv[8]

# experiment parameters
L = 100.0e3                       # domain is (-L,L)
SMBlist = [0.0, -2.5e-7, 1.0e-7]  # m s-1; values of aconst used in experiments
aposfrac = 0.75                   # fraction of domain on which positive SMB is applied
Nsamples = 1000                   # number of samples when evaluating minimal ratios

# solution method
zeroheight = 'indices'  # how should StokesExtrude handle zero-height columns;
                        #   alternative is 'bounds', but it seems to do poorly?
fssa = True             # use Lofgren et al (2022) FSSA technique in Stokes solve
theta_fssa = 1.0        #   with this theta value

# Stokes parameters
pp = (1.0 / nglen) + 1.0
Dtyp = 1.0 / secpera      # = 1 a-1; strain rate scale
mu0 = 0.0001 * Dtyp**2.0  # viscosity regularization

# set up bm = basemesh once
bm = IntervalMesh(mx, -L, L)
P1bm = FunctionSpace(bm, 'P', 1)
xbm = bm.coordinates.dat.data_ro

# bed and initial geometry
assert bed in bedtypes
print(f"Halfar t0 = {t0 / secpera:.3f} a")
b_np, s_initial_np = generategeometry(xbm, t=t0, bed=bed)  # get numpy arrays
b = Function(P1bm, name='bed elevation (m)')
b.dat.data[:] = b_np
s_initial = Function(P1bm)
s_initial.dat.data[:] = s_initial_np

# surface mass balance (SMB) function
a = Function(P1bm, name='surface mass balance (m s-1)')

# create the extruded mesh, but leave z coordinate at default
se = StokesExtrude(bm, mz=mz, htol=1.0)
P1 = FunctionSpace(se.mesh, 'P', 1)
x, _ = SpatialCoordinate(se.mesh)

# set up Stokes problem
se.mixed_TaylorHood()
#se.mixed_PkDG(kp=0) # = P2xDG0; seems to NOT be better; not sure about P2xDG1
se.body_force(Constant((0.0, - rho * g)))
se.dirichlet((1,2), Constant((0.0, 0.0)))      # consequences if ice advances to margin
se.dirichlet(('bottom',), Constant((0.0, 0.0)))
params = SolverParams['newton']
params.update(SolverParams['mumps'])
#params.update({'snes_monitor': None})
params.update({'snes_converged_reason': None})
params.update({'snes_atol': 1.0e-2})
params.update({'snes_linesearch_type': 'bt'})  # helps with non-flat beds, it seems

# initialize surface elevation state variable
s = Function(P1bm, name='surface elevation (m)')
s.interpolate(conditional(s_initial < b, b, s_initial))

# set up for semi-implicit method, documented in paper, using old velocity in weak form
sibcs = [DirichletBC(P1bm, b.dat.data_ro[0], 1),
         DirichletBC(P1bm, b.dat.data_ro[-1], 2)]
siv = TestFunction(P1bm)
sisold = Function(P1bm)
siP2Vbm = VectorFunctionSpace(bm, 'CG', 2, dim=2)
siubm = Function(siP2Vbm)  # surface velocity
siparams = {#"snes_monitor": None,
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
            "pc_factor_mat_solver_type": "mumps"}

# weak form for semi-implicit; UN-regularized
siF = dt * Phi(s, siubm, siv) * dx + inner(s - (sisold + dt * a), siv) * dx
siproblem = NonlinearVariationalProblem(siF, s, sibcs)
sisolver = NonlinearVariationalSolver(siproblem, solver_parameters=siparams,
                                      options_prefix="step")
siub = Function(P1bm).interpolate(Constant(PETSc.INFINITY))

# set up for livefigure() and snapsfigure()
if writepng:
    printpar(f'creating root directory {dirroot} for image files ...')
    mkdir(dirroot)
    snaps = [s.copy(deepcopy=True),]

# outer surface mass balance (SMB) loop
_slist = []
for aconst in SMBlist:
    # describe run
    printpar(f'using aconst = {aconst:.3e} m/s constant value of SMB ...')
    printpar(f'doing N = {Nsteps} steps of dt = {dt/secpera:.3f} a and saving states ...')
    printpar(f'  solving 2D Stokes + SKE on {mx} x {mz} extruded mesh over {bed} bed')
    printpar(f'  dimensions: n_u = {se.V.dim()}, n_p = {se.W.dim()}')
    # set up directory and open file (if wanted)
    afrag = 'aneg' if aconst < 0.0 else ('apos' if aconst > 0.0 else 'azero')
    if writepng:
        outdirname = dirroot + afrag + '/'
        printpar(f'  creating directory {outdirname} for image files ...')
        mkdir(outdirname)
    if writepvd:
        pvdfilename = pvdroot + '_' + afrag + '.pvd'
        printpar(f'  opening {pvdfilename} ...')
        outfile = VTKFile(pvdfilename)
    # reset for time-stepping
    if aconst > 0.0:
        a.dat.data[:] = 0.0
        a.dat.data[abs(xbm) < aposfrac * L] = aconst
    else:
        a.dat.data[:] = aconst
    s.interpolate(conditional(s_initial < b, b, s_initial))
    t = 0.0

    # inner time-stepping loop
    for n in range(Nsteps):
        # start with reporting
        if n == 0:
            geometryreport(bm, 0, t, s, b, Lsc=L)
            if writepng:
                livefigure(bm, b, s, t, fname=f'{outdirname}t{t/secpera:010.3f}.png')

        # set geometry (z coordinate) of extruded mesh
        se.reset_elevations(b, s)
        P1R = FunctionSpace(se.mesh, 'P', 1, vfamily='R', vdegree=0)
        sR = Function(P1R)
        sR.dat.data[:] = s.dat.data_ro

        # for saving with current geometry: solve Stokes on extruded mesh and extract surface trace
        stokesF = form_stokes(se, sR, pp=pp, mu0=mu0, fssa=False)
        u, p = se.solve(F=stokesF, par=params, zeroheight=zeroheight)
        ubm = trace_vector_to_p2(bm, se.mesh, u)  # surface velocity (m s-1)

        # optionally write t-dependent .pvd with 2D fields
        if writepvd:
            u.rename('velocity (m s-1)')
            p.rename('pressure (Pa)')
            nu, nueps = effective_viscosity(u, P1, pp=pp, mu0=mu0)
            phydro = p_hydrostatic(se, sR, P1)
            pdiff = Function(P1).interpolate(p - phydro)
            pdiff.rename('pdiff = phydro - p (Pa)')
            outfile.write(u, p, nu, nueps, pdiff, time=t)

        # save surface elevation and surface velocity into list for ratio evals
        _slist.append({'t': t,
                       's': s.copy(deepcopy=True),
                       'us': ubm.copy(deepcopy=True)})

        # for semi-implicit step, solve Stokes on extruded mesh *WITH FSSA* and extract surface trace
        stokesF = form_stokes(se, sR, pp=pp, mu0=mu0,
                              fssa=fssa, theta_fssa=theta_fssa, dt_fssa=dt, smb_fssa=a)
        u, p = se.solve(F=stokesF, par=params, zeroheight=zeroheight)
        ubm = trace_vector_to_p2(bm, se.mesh, u)  # surface velocity (m s-1)

        # time step of VI problem; semi-implicit: solve VI problem with surface velocity from old surface elevation
        sisold.dat.data[:] = s.dat.data_ro
        siubm.dat.data[:] = ubm.dat.data_ro
        sisolver.solve(bounds=(b, siub))
        t += dt

        # end of step reporting
        geometryreport(bm, n + 1, t, s, b, Lsc=L)
        if writepng:
            tfilename = f'{outdirname}t{t/secpera:010.3f}.png'
            wh = (bed == 'flat' and aconst == 0.0 and n + 1 == Nsteps)
            livefigure(bm, b, s, t, fname=tfilename, writehalfar=wh)
            if n + 1 == int(round(0.7 * Nsteps)): # reliable if Nsteps is divisible by 10
                snaps.append(s.copy(deepcopy=True))

    if writepng:
        printpar(f'  finished writing to {outdirname}')
    if writepvd:
        printpar(f'  finished writing to {pvdfilename}')

if writepng:
    snapsname = dirroot + 'snaps.png'
    snapsfigure(bm, b, snaps, fname=snapsname)
    printpar(f'  finished writing to {snapsname}')

# process _slist from all three SMB cases, with and without regularization
# writes .csv files with data which allows reconstruction of ratios
eps = [0.0, 0.1]
epsstr = ['NOREG', 'REG__']
for j in range(2):
    root = dirroot + epsstr[j] + '/'
    mkdir(root)
    dfilename = root + 'data.csv'
    maxcont, rats = sampleratios(dfilename, _slist, bm, b, N=Nsamples, Lsc=L, epsreg=eps[j])
    if j == 0:
        printpar(f'  max continuity ratio:               {maxcont:.3e}')
    pos = rats[rats > 0.0]
    assert len(pos) > 0
    pmin = min(pos)
    pmed = np.median(pos)
    printpar(f'  pos coercivity ratio {epsstr[j]} min:           {pmin:.3e}')
    printpar(f'                       {epsstr[j]} median:        {pmed:.3e}')
    nonpos = rats[rats <= 0.0]
    if len(nonpos) > 0:
        npmin, npmed, npf = min(nonpos), np.median(nonpos), len(nonpos) / len(rats)
        printpar(f'  non-pos coercivity ratio {epsstr[j]} min:       {npmin:.3e}')
        printpar(f'                           {epsstr[j]} median:    {npmed:.3e}')
        printpar(f'                           {epsstr[j]} fraction:  {npf:.4f}')
        with open(ratiosfile, 'a') as rfile:
            rfile.write(f'{epsstr[j]}: {maxcont:.3e}, {pmin:.3e}, {pmed:.3e}, {npmin:.3e}, {npmed:.3e}, {npf:.4f}\n')
    else:
        with open(ratiosfile, 'a') as rfile:
            rfile.write(f'{epsstr[j]}: {maxcont:.3e}, {pmin:.3e}, {pmed:.3e}, N/A, N/A, 0.0000\n')
