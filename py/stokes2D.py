#!/usr/bin/env python3

# SHOWS EXTRUDED ORDERING:
# ./stokes2D.py -s_ksp_view_mat draw -draw_pause -1 -draw_size 1000,1000 -nintervals 10 -layers 4

from firedrake import *

import argparse
parser = argparse.ArgumentParser(description=
'''
Generate 2D mesh from Halfar (1981) used extrusion of an equally-spaced
interval mesh.  Solve Laplace equation for reference domain scheme for SMB.
''')
parser.add_argument('-H0', type=float, default=5000.0, metavar='X',
                    help='center height in m of ice sheet (default=5000)')
parser.add_argument('-Href', type=float, default=500.0, metavar='X',
                    help='minimum thickness in m of reference domain (default=500)')
parser.add_argument('-L', type=float, default=60.0e3, metavar='X',
                    help='half-width in m of computational domain (default=60e3)')
parser.add_argument('-layers', type=int, default=5, metavar='N',
                    help='number of layers in each column (default=5)')
parser.add_argument('-nintervals', type=int, default=30, metavar='N',
                    help='number of (equal) subintervals in computational domain (default=30)')
parser.add_argument('-R0', type=float, default=50.0e3, metavar='X',
                    help='half-width in m of ice sheet (default=50e3)')
parser.add_argument('-refine', type=int, default=-1, metavar='X',
                    help='number of mesh refinement levels (e.g. for GMG)')
parser.add_argument('-o', metavar='NAME', type=str, default='',
                    help='output file name ending with .pvd')
args, unknown = parser.parse_known_args()

# physical constants
g = 9.81             # m s-2; constants in SI units
rho = 910.0          # kg
secpera = 31556926.0
n = 3.0
A = 1.0e-16/secpera
Gamma = 2.0 * A * (rho * g)**3.0 / 5.0   # see Bueler et al (2005)

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
    Vc = kmesh.coordinates.function_space()
    f = Function(Vc).interpolate(as_vector([x,Hlimited*y]))
    kmesh.coordinates.assign(f)

# report on generated geometry and fine mesh
dxelem = 2.0 * args.L / (mx-1)
dyrefelem = args.Href / (my-1)
PETSc.Sys.Print('generating 2D Halfar geometry on interval [%.2f,%.2f] km,'
                % (-args.L/1000.0,args.L/1000.0))
PETSc.Sys.Print('    with H0=%.2f m and R0=%.2f km, at t0=%.5f a,'
                % (args.H0,args.R0/1000.0,t0/secpera))
PETSc.Sys.Print('    as %d x %d element quadrilateral extruded (fine) mesh, limited at Href=%.2f m,'
                % (mx-1,my-1,args.Href))
PETSc.Sys.Print('    reference element dimensions: dx=%.2f m, dy=%.2f m, ratio=%.5f'
                % (dxelem,dyrefelem,dyrefelem/dxelem))

# space in which to solve Laplace equation (nabla^2 u = 0) with Dirichlet bcs
V = FunctionSpace(mesh, "CG", 1)

# simulate surface kinematical condition value for top; FIXME actual!
dt = secpera  # 1 year time steps
a = Constant(0.0)
x,y = SpatialCoordinate(mesh)
smbref = conditional(y > args.Href, dt*a, dt*a - args.Href)  # FIXME
bcs = [DirichletBC(V, smbref, 'top'),
       DirichletBC(V, Constant(0.0), 'bottom')]

# Laplace equation weak form
u = Function(V, name='u')  # initialized to zero here
v = TestFunction(V)
F = dot(grad(u), grad(v)) * dx

# Solve system as though it is nonlinear:  F(u) = 0
solve(F == 0, u, bcs=bcs, options_prefix = 's')  # FIXME solver?

# output ParaView-readable file
if args.o:
    PETSc.Sys.Print('writing ice geometry and solution to %s ...' % args.o)
    File(args.o).write(u)

