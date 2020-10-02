#!/usr/bin/env python3

# TODO:
#   * initialize with u=(SIA velocity), p=(hydrostatic) and c=0
#   * option -sialaps N: do SIA evals N times and quit; for timing; defines work unit
#   * implement displacement stretching scheme

from firedrake import *

import argparse
parser = argparse.ArgumentParser(description='''
Solve coupled Glen-Stokes plus surface kinematical equation (SKE) problem for
an ice sheet on a flat bed.  Generates 2D mesh from Halfar (1981) by extrusion
of an equally-spaced interval mesh, thus the elements are quadrilaterals.
A reference domain with a minimum thickness is used.  For velocity u,
pressure p, and scalar vertical displacement c we set up a nonlinear system
of three equations corresponding to a single time step of -dta years:
   stress balance:       F_1(u,p)   = 0    FIXME: also c when stretched
   incompressibility:    F_2(u)     = 0    FIXME: also c when stretched
   Laplace/SKE:          F_3(u,c)   = 0
The last equation is Laplace's equation for c in the domain interior but it
is coupled to the first two through the top boundary condition which enforces
the SKE.  The mixed space consists of (u,p) in Q2 x dQ0 for the Stokes problem
and c in Q1 for displacement.  The default solver is multiplicative fieldsplit
between the Stokes (u,p) block and the c block, with Stokes solved by Schur
lower fieldsplit with selfp preconditioning on the Schur block.  The diagonal
blocks are solved by AMG (-gamg).
''',add_help=False)
parser.add_argument('-dirichletsmb', action='store_true', default=False,
                    help='apply simplified SMB condition on top of reference domain')
parser.add_argument('-dta', type=float, default=1.0, metavar='X',
                    help='length of time step in years')
parser.add_argument('-Dtyp', type=float, default=2.0, metavar='X',
                    help='typical strain rate in "+(eps Dtyp)^2" (default=2.0 a-1)')
parser.add_argument('-eps', type=float, default=0.01, metavar='X',
                    help='to regularize viscosity by "+(eps Dtyp)^2" (default=0.01)')
parser.add_argument('-H0', type=float, default=3000.0, metavar='X',
                    help='center height in m of ice sheet (default=3000)')
parser.add_argument('-Href', type=float, default=500.0, metavar='X',
                    help='minimum thickness in m of reference domain (default=500)')
parser.add_argument('-L', type=float, default=60.0e3, metavar='X',
                    help='half-width in m of computational domain (default=60e3)')
parser.add_argument('-layers', type=int, default=10, metavar='N',
                    help='number of layers in each column (default=10)')
parser.add_argument('-linear', action='store_true', default=False,
                    help='use linear, trivialized Stokes problem')
parser.add_argument('-nintervals', type=int, default=30, metavar='N',
                    help='number of (equal) subintervals in computational domain (default=30)')
parser.add_argument('-o', metavar='NAME', type=str, default='',
                    help='output file name ending with .pvd')
parser.add_argument('-R0', type=float, default=50.0e3, metavar='X',
                    help='half-width in m of ice sheet (default=50e3)')
parser.add_argument('-refine', type=int, default=-1, metavar='X',
                    help='number of mesh refinement levels (e.g. for GMG)')
parser.add_argument('-sia', action='store_true', default=False,
                    help='use a coupled weak form corresponding to the SIA problem')
parser.add_argument('-stokes2Dhelp', action='store_true', default=False,
                    help='help for stokes2D.py options')
args, unknown = parser.parse_known_args()
if args.stokes2Dhelp:
    parser.print_help()

# physical constants
g = 9.81             # m s-2; constants in SI units
rho = 910.0          # kg
secpera = 31556926.0
n = 3.0              # warning: this value is hardwired in some places below
A3 = 1.0e-16/secpera

# computed constants and regularization parameters
Gamma = 2.0 * A3 * (rho * g)**3.0 / 5.0   # see Bueler et al (2005)
B3 = A3**(-1.0/3.0)                       # Pa s(1/3);  ice hardness
Dtyp = args.Dtyp / secpera
dt = args.dta * secpera

# Halfar (1981) solution
alpha = 1.0/11.0
t0 = (alpha/Gamma) * (7.0/4.0)**3.0 * (args.R0**4.0 / args.H0**7.0)
def halfar2d(x):
    absx = sqrt(pow(x,2))
    inpow = (n+1) / n
    outpow = n / (2*n+1)
    inside = max_value(0.0, 1.0 - pow(absx / args.R0,inpow))
    return args.H0 * pow(inside,outpow)

# set up base mesh, as hierarchy if requested
mx = args.nintervals + 1
base_mesh = IntervalMesh(args.nintervals, length_or_left=-args.L, right=args.L)
if args.refine > 0:
    base_hierarchy = MeshHierarchy(base_mesh, args.refine)
    base_mesh = base_hierarchy[-1]     # the fine mesh
    mx = (mx-1) * 2**args.refine + 1

# set up extruded mesh, as hierarchy if requested
my = args.layers + 1
temporary_height = 1.0
if args.refine > 0:
    hierarchy = ExtrudedMeshHierarchy(base_hierarchy, temporary_height, base_layer=args.layers)
    mesh = hierarchy[-1]     # the fine mesh
    my = (my-1) * 2**args.refine + 1
else:
    mesh = ExtrudedMesh(base_mesh, layers=args.layers, layer_height=temporary_height/args.layers)

# note: At this point,  N = mx my  is the number of nodes in the extruded mesh,
#       with mx-1 elements (subintervals) in the horizontal and my-1 elements
#       in the vertical.

# deform y coordinate, in each level of hierarchy, to match Halfar solution, but limited at Href
if args.refine > 0:
    hierlevs = args.refine + 1
else:
    hierlevs = 1
for k in range(hierlevs):
    if args.refine > 0:
        kmesh = hierarchy[k]
    else:
        kmesh = mesh
    x,y = SpatialCoordinate(kmesh)
    Hinitial = halfar2d(x)
    Hlimited = max_value(args.Href, Hinitial)
    Vcoord = kmesh.coordinates.function_space()
    f = Function(Vcoord).interpolate(as_vector([x,Hlimited*y]))
    kmesh.coordinates.assign(f)

# fine mesh coordinates
x,y = SpatialCoordinate(mesh)

# report on generated geometry and fine mesh
dxelem = 2.0 * args.L / (mx-1)
dyrefelem = args.Href / (my-1)
PETSc.Sys.Print('initial condition: 2D Halfar with H0=%.2f m and R0=%.2f km, at t0=%.5f a'
                % (args.H0,args.R0/1000.0,t0/secpera))
PETSc.Sys.Print('domain: interval [%.2f,%.2f] km extruded to initial, limited at Href=%.2f m'
                % (-args.L/1000.0,args.L/1000.0,args.Href))
PETSc.Sys.Print('mesh: %d x %d element quadrilateral (fine) mesh'
                % (mx-1,my-1))
PETSc.Sys.Print('element dimensions: dx=%.2f m, dy_min=%.2f m, ratio=%.5f'
                % (dxelem,dyrefelem,dyrefelem/dxelem))
PETSc.Sys.Print('computing one time step dt=%.5f a ...' % args.dta)

# mixed spaces:  Q2 x dQ0 for Stokes problem, and Q1 for displacement
Vu = VectorFunctionSpace(mesh, 'CG', degree=2)  # velocity  u = (u_0(x,y),u_1(x,y))
Vp = FunctionSpace(mesh, 'DG', degree=0)        # pressure  p(x,y)
Vc = FunctionSpace(mesh, 'CG', degree=1)        # displacement  c(x,y)
Z = Vu * Vp * Vc

# trial and test functions
upc = Function(Z)
u,p,c = split(upc)
v,q,e = TestFunctions(Z)

# define the nonlinear weak form F(u,p,c;v,q,e)
Du = 0.5 * (grad(u)+grad(u).T)
Dv = 0.5 * (grad(v)+grad(v).T)
f_body = Constant((0.0, - rho * g))
# FIXME stretching in F; couples (u,p) and c problems
if args.linear:   # linear Stokes with viscosity = 1.0
    tau = 2.0 * Du
else:
    if args.sia:  # FIXME not sure if this is SIA!  TEST with -s_mat_type aij -s_pc_type svd -s_ksp_type preonly
        Du = as_matrix([[0, 0.5*u[0].dx(1)], [0, 0]])
        Dv = as_matrix([[v[0].dx(1), 0], [0, 0]])
        #FIXME: add this?  DirichletBC(Z.sub(1), Constant(0.0), 'top'),         # SIA: zero pressure on top
    # n=3 Glen law Stokes
    Du2 = 0.5 * inner(Du, Du) + (args.eps * Dtyp)**2.0
    tau = B3 * Du2**(-1.0/3.0) * Du
F =  inner(tau, Dv) * dx \
     + ( - p * div(v) - div(u) * q - inner(f_body,v) ) * dx \
     + inner(grad(c),grad(e)) * dx

# construct equation for surface kinematical equation (SKE) boundary condition
a = Constant(0.0) # correct for Halfar
if args.dirichletsmb:
    # artificial case: apply solution-independent smb as Dirichlet
    smb = a
    smbref = conditional(y > args.Href, dt * smb, dt * smb - args.Href)  # FIXME: try "y>0"
    bctop = DirichletBC(Z.sub(2), smbref, 'top')
else:
    # in default coupled case, SMB is an equation
    smb = a - u[0] * y.dx(0) + u[1]  #  a - u dh/dx + w
    smbref = conditional(y > args.Href, dt * smb, dt * smb - args.Href)  # FIXME: try "y>0"
    Fsmb = (c - smbref) * e * ds_t
    bctop = EquationBC(Fsmb == 0, upc, 'top', V=Z.sub(2))

# list boundary conditions
bcs = [DirichletBC(Z.sub(0), Constant((0.0, 0.0)), 'bottom'),  # zero velocity on bottom
       DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1,2)),     # zero velocity on the ends
       DirichletBC(Z.sub(2), Constant(0.0), 'bottom'),         # zero displacement on the bottom
       bctop]                                                  # SKE equation on the top

# solver parameters
parameters = {'mat_type': 'aij',
              'pc_type': 'fieldsplit',
              # (u,p)-(u,p) and c-c diagonal blocks are coupled by (lower) c-u block
              # FIXME: reconsider when stretching adds u-c and p-c blocks
              'pc_fieldsplit_type': 'multiplicative',  # 'additive': more linear iters
              'pc_fieldsplit_0_fields': '0,1',
              'pc_fieldsplit_1_fields': '2',
              # schur fieldsplit for (u,p)-(u,p) block
              'fieldsplit_0_ksp_type': 'preonly',
              'fieldsplit_0_pc_type': 'fieldsplit',
              'fieldsplit_0_pc_fieldsplit_type': 'schur',
              'fieldsplit_0_pc_fieldsplit_schur_fact_type': 'lower',
              # selfp seems to be faster than a Mass object
              'fieldsplit_0_pc_fieldsplit_schur_precondition': 'selfp',
              # AMG on the u-u block; mg works but slower
              'fieldsplit_0_fieldsplit_0_pc_type': 'gamg',
              'fieldsplit_0_fieldsplit_1_pc_type': 'jacobi',
              'fieldsplit_0_fieldsplit_1_pc_jacobi_type': 'diagonal',
              'fieldsplit_1_ksp_type': 'preonly',
              # AMG on the c-c block; mg fails with zero row msg; hypre (w/o tuning) seems slower
              # classical few iters and faster than agg (but grid complexity better for agg)
              'fieldsplit_1_pc_type': 'gamg',
              'fieldsplit_1_pc_gamg_type': 'classical',
              'fieldsplit_1_pc_gamg_square_graph': '1'}

# solve system as though it is nonlinear:  F(u) = 0
solve(F == 0, upc, bcs=bcs, options_prefix = 's',
      solver_parameters=parameters)

# save ParaView-readable file
if args.o:
    PETSc.Sys.Print('writing ice geometry and solution (u,p,c) to %s ...' % args.o)
    u,p,c = upc.split()
    u.rename('velocity')
    p.rename('pressure')
    c.rename('displacement')
    File(args.o).write(u,p,c)

