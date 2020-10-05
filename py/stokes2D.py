#!/usr/bin/env python3

# TODO:
#   * initialize with u=(SIA velocity), p=(hydrostatic) and c=0
#   * option -sialaps N: do SIA evals N times and quit; for timing; defines work unit
#   * implement displacement stretching scheme
#   * "aerogel mush" above current iterate surface in Href

# example: runs in about a minute with 5/2 element ratio and N=1.6e5
# timer ./stokes2D.py -dta 0.1 -s_snes_converged_reason -s_ksp_converged_reason -s_snes_rtol 1.0e-4 -mx 960 -refine 1 -savetau -o foo.pvd

from firedrake import *
import sys
from iceconstants import secpera,g,rho,n,A3,B3,Gamma
from halfar import halfar2d, halfar3d

import argparse
parser = argparse.ArgumentParser(description='''
Solve coupled Glen-Stokes plus surface kinematical equation (SKE) for
an ice sheet.  Generates flat-bed 2D mesh from Halfar (1981) by extrusion
of an equally-spaced interval mesh, thus the elements are quadrilaterals.
A reference domain with a minimum thickness is used.  For velocity u,
pressure p, and scalar vertical displacement c we set up a nonlinear system
of three equations corresponding to a single time step of -dta years:
   stress balance:       F_1(u,p)   = 0    FIXME: also c when stretched
   incompressibility:    F_2(u)     = 0    FIXME: also c when stretched
   Laplace/SKE:          F_3(u,c)   = 0
The last equation is Laplace's equation for c in the domain interior but it
is coupled to the first two through the top boundary condition which enforces
the SKE.  The mixed space consists of (u,p) in Q2 x Q1 for the Stokes problem
and c in Q1 for displacement.  The default solver is multiplicative fieldsplit
between the Stokes (u,p) block and the c block.  The Stokes block is solved by
Schur lower fieldsplit with selfp preconditioning on the Schur block.  The
diagonal blocks are solved by AMG (-gamg).''',
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 add_help=False)
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
parser.add_argument('-Href', type=float, default=200.0, metavar='X',
                    help='minimum thickness in m of reference domain (default=200)')
parser.add_argument('-L', type=float, default=60.0e3, metavar='X',
                    help='half-width in m of computational domain (default=60e3)')
parser.add_argument('-layers', type=int, default=4, metavar='N',
                    help='number of layers in each vertical column (default=4)')
parser.add_argument('-linear', action='store_true', default=False,
                    help='use linear, trivialized Stokes problem')
parser.add_argument('-mx', type=int, default=30, metavar='N',
                    help='number of equal subintervals in x-direction (default=30)')
parser.add_argument('-my', type=int, default=-1, metavar='N',
                    help='solve in 3D with this number of equal subintervals in y-direction (default=30)')
parser.add_argument('-o', metavar='FILE.pvd', type=str, default='',
                    help='save to output file name ending with .pvd')
parser.add_argument('-R0', type=float, default=50.0e3, metavar='X',
                    help='half-width in m of ice sheet (default=50e3)')
parser.add_argument('-refine', type=int, default=-1, metavar='N',
                    help='number of mesh refinement levels (e.g. for GMG)')
parser.add_argument('-savetau', action='store_true',
                    help='save deviatoric stress tensor to output file', default=False)
parser.add_argument('-sia', action='store_true', default=False,
                    help='use a coupled weak form corresponding to the SIA problem')
parser.add_argument('-spectralvert', type=int, default=0, metavar='N',
                    help='stages for p-refinement in vertical; use 0,1,2,3 only  (default=0)')
parser.add_argument('-stokes2Dhelp', action='store_true', default=False,
                    help='print help for stokes2D.py and quit')
args, unknown = parser.parse_known_args()
if args.stokes2Dhelp:
    parser.print_help()
    sys.exit(0)

# regularization constant
Dtyp = args.Dtyp / secpera

# timestep  FIXME need time-stepping loop
dt = args.dta * secpera

# set up base mesh, as hierarchy if requested
mx = args.mx
if args.my > 0:
    my = args.my
    base_mesh = RectangleMesh(mx, my, 2.0*args.L, 2.0*args.L, quadrilateral=True)
    base_mesh.coordinates.dat.data[:, 0] -= args.L
    base_mesh.coordinates.dat.data[:, 1] -= args.L
    PETSc.Sys.Print('3D mode coarse base mesh: %d x %d element quadrilaterals over %d processes'
                % (mx,my,base_mesh.comm.size))
else:
    base_mesh = IntervalMesh(mx, length_or_left=-args.L, right=args.L)
    PETSc.Sys.Print('2D mode coarse base mesh: %d element intervals over %d processes'
                % (mx,base_mesh.comm.size))

if args.refine > 0:
    base_hierarchy = MeshHierarchy(base_mesh, args.refine)
    base_mesh = base_hierarchy[-1]     # the fine mesh
    mx *= 2**args.refine
    if args.my > 0:
        my *= 2**args.refine

# set up extruded mesh, as hierarchy if requested
mz = args.layers
temporary_height = 1.0
if args.refine > 0:
    hierarchy = ExtrudedMeshHierarchy(base_hierarchy, temporary_height, base_layer=args.layers)
    mesh = hierarchy[-1]     # the fine mesh
    mz *= 2**args.refine
else:
    mesh = ExtrudedMesh(base_mesh, layers=args.layers, layer_height=temporary_height/args.layers)

# note: in 2D,  mx mz     is the number of elements in the extruded mesh
#       in 3D,  mx my mz  ...

# deform z coordinate, in each level of hierarchy, to match Halfar solution, but limited at Href
if args.refine > 0:
    hierlevs = args.refine + 1
else:
    hierlevs = 1
for k in range(hierlevs):
    if args.refine > 0:
        kmesh = hierarchy[k]
    else:
        kmesh = mesh
    if args.my > 0:
        x,y,z = SpatialCoordinate(kmesh)
    else:
        x,z = SpatialCoordinate(kmesh)
    t0, Hinitial = halfar2d(x,R0=args.R0,H0=args.H0)
    Hlimited = max_value(args.Href, Hinitial)
    Vcoord = kmesh.coordinates.function_space()
    if args.my > 0:
        f = Function(Vcoord).interpolate(as_vector([x,y,Hlimited*z]))
    else:
        f = Function(Vcoord).interpolate(as_vector([x,Hlimited*z]))
    kmesh.coordinates.assign(f)

# fine mesh coordinates
if args.my > 0:
    x,y,z = SpatialCoordinate(mesh)
else:
    x,z = SpatialCoordinate(mesh)

# optional: p-refinement in vertical for (u,p); args.spectralvert in {0,1,2,3}
degreexz = [(2,1),(3,2),(4,3),(5,4)]
zudeg,zpdeg = degreexz[args.spectralvert]

#FIXME from here: to allow 3D, xuE is either interval or unitsquare (same for xpE, xcE)

# construct component spaces by explicitly applying TensorProductElement()
# Q2 for velocity u = (u_0(x,y),u_1(x,y))
xuE = FiniteElement('CG',interval,2)
zuE = FiniteElement('CG',interval,zudeg)
uE = TensorProductElement(xuE,zuE)
Vu = VectorFunctionSpace(mesh, uE)
# Q1 for pressure p(x,y)
xpE = FiniteElement('CG',interval,1)
zpE = FiniteElement('CG',interval,zpdeg)
pE = TensorProductElement(xpE,zpE)
Vp = FunctionSpace(mesh, pE)
# Q1 for displacement c(x,y)
xcE = FiniteElement('CG',interval,1)
zcE = FiniteElement('CG',interval,1)  # consider raising to 2: field "looks better?"
cE = TensorProductElement(xcE,zcE)
Vc = FunctionSpace(mesh, cE)

# construct mixed space
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
        Du = as_matrix([[0, 0.5*u[0].dx(1)], [0, 0]])  # FIXME 2D only
        Dv = as_matrix([[v[0].dx(1), 0], [0, 0]])
        #FIXME: add this?  DirichletBC(Z.sub(1), Constant(0.0), 'top'),         # SIA: zero pressure on top
    # n=3 Glen law Stokes
    Du2 = 0.5 * inner(Du, Du) + (args.eps * Dtyp)**2.0
    tau = B3 * Du2**(-1.0/3.0) * Du
F = inner(tau, Dv) * dx \
    + ( - p * div(v) - div(u) * q - inner(f_body,v) ) * dx \
    + inner(grad(c),grad(e)) * dx

# construct equation for surface kinematical equation (SKE) boundary condition
a = Constant(0.0) # correct for Halfar
if args.dirichletsmb:
    # artificial case: apply solution-independent smb as Dirichlet
    smb = a
    smbref = conditional(z > args.Href, dt * smb, dt * smb - args.Href)  # FIXME: try "y>0"
    bctop = DirichletBC(Z.sub(2), smbref, 'top')
else:
    # in default coupled case, SMB is an equation
    smb = a - u[0] * z.dx(0) + u[1]  #  a - u dh/dx + w
    smbref = conditional(z > args.Href, dt * smb, dt * smb - args.Href)  # FIXME: try "y>0"
    Fsmb = (c - smbref) * e * ds_t
    bctop = EquationBC(Fsmb == 0, upc, 'top', V=Z.sub(2))

# list boundary conditions
bcs = [DirichletBC(Z.sub(0), Constant((0.0, 0.0)), 'bottom'),  # zero velocity on bottom
       DirichletBC(Z.sub(0), Constant((0.0, 0.0)), (1,2)),     # zero velocity on the ends
       DirichletBC(Z.sub(2), Constant(0.0), 'bottom'),         # zero displacement on the bottom
       bctop]                                                  # SKE equation on the top

# report on generated geometry and fine mesh
dxelem = 2.0 * args.L / mx
dzrefelem = args.Href / mz
PETSc.Sys.Print('initial condition: 2D Halfar with H0=%.2f m and R0=%.3f km, at t0=%.5f a'
                % (args.H0,args.R0/1000.0,t0/secpera))
PETSc.Sys.Print('domain: [%.2f,%.2f] km extruded and limited at Href=%.2f m'
                % (-args.L/1000.0,args.L/1000.0,args.Href))
if args.my > 0:
    PETSc.Sys.Print('fine 3D extruded mesh: %d x %d element quadrilateral'
                    % (mx,my,mz))
else:
    PETSc.Sys.Print('fine 2D extruded mesh: %d x %d element quadrilateral'
                    % (mx,mz))
PETSc.Sys.Print('element dimensions: dx=%.2f m, dz_min=%.2f m, ratio=%.1f'
                % (dxelem,dzrefelem,dxelem/dzrefelem))
n_u,n_p,n_c,N = Vu.dim(),Vp.dim(),Vc.dim(),Z.dim()
PETSc.Sys.Print('vector space dimensions : n_u=%d, n_p=%d, n_c=%d ... N=%d' \
                % (n_u,n_p,n_c,N))
PETSc.Sys.Print('computing one time step dt=%.5f a ...' % args.dta)

# solver parameters; many are defaults which are deliberately made explicit here
parameters = {'mat_type': 'aij',
              'ksp_type': 'gmres',
              'ksp_pc_side': 'left',
              # (u,p)-(u,p) and c-c diagonal blocks are coupled by (lower) c-u block
              # FIXME: reconsider when stretching adds u-c and p-c blocks
              'pc_type': 'fieldsplit',
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
              'fieldsplit_0_fieldsplit_0_ksp_type': 'preonly',
              'fieldsplit_0_fieldsplit_0_pc_type': 'lu',
              #'fieldsplit_0_fieldsplit_0_pc_type': 'gamg',
              #'fieldsplit_0_fieldsplit_0_pc_gamg_type': 'agg',
              #'fieldsplit_0_fieldsplit_0_mg_levels_ksp_type': 'chebyshev',
              #'fieldsplit_0_fieldsplit_0_mg_levels_pc_type': 'sor',
              'fieldsplit_0_fieldsplit_1_pc_type': 'jacobi',
              'fieldsplit_0_fieldsplit_1_pc_jacobi_type': 'diagonal',
              # AMG on the c-c block; mg fails with zero row msg; hypre (w/o tuning) seems slower
              # classical few iters and faster than agg (but grid complexity better for agg)
              'fieldsplit_1_ksp_type': 'preonly',
              'fieldsplit_1_pc_type': 'lu'}
              #'fieldsplit_1_pc_type': 'gamg',
              #'fieldsplit_1_pc_gamg_type': 'classical',
              #'fieldsplit_1_pc_gamg_square_graph': '1'}

# solve system as though it is nonlinear:  F(u) = 0
solve(F == 0, upc, bcs=bcs, options_prefix = 's',
      solver_parameters=parameters)

# save ParaView-readable file
if args.o:
    written = 'u,p,c'
    if mesh.comm.size > 1:
         written += ',rank'
    if args.savetau:
         written += ',tau'
    PETSc.Sys.Print('writing solution variables (%s) to output file %s ... ' % (written,args.o))
    u,p,c = upc.split()
    u.rename('velocity')
    p.rename('pressure')
    c.rename('displacement')
    if mesh.comm.size > 1:
        # integer-valued element-wise process rank
        rank = Function(FunctionSpace(mesh,'DG',0))
        rank.dat.data[:] = mesh.comm.rank
        rank.rename('rank')
    if args.savetau:
        # tensor-valued deviatoric stress tau
        TQ1 = TensorFunctionSpace(mesh, 'CG', 1)
        Du = Function(TQ1).interpolate(0.5 * (grad(u)+grad(u).T))
        Du2 = Function(Vp).interpolate(0.5 * inner(Du, Du) + (args.eps * args.Dtyp)**2.0)
        tau = Function(TQ1).interpolate(B3 * Du2**(-1.0/3.0) * Du)
        tau.rename('tau')
    if mesh.comm.size > 1 and args.savetau:
        File(args.o).write(u,p,c,rank,tau)
    elif mesh.comm.size > 1:
        File(args.o).write(u,p,c,rank)
    elif args.savetau:
        File(args.o).write(u,p,c,tau)
    else:
        File(args.o).write(u,p,c)

