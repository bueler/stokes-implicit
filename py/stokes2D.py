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
                    help='minimum thickness in m of reference domain (default=200)')
parser.add_argument('-L', type=float, default=60.0e3, metavar='X',
                    help='half-width in m of computational domain (default=60e3)')
parser.add_argument('-layers', type=int, default=10, metavar='N',
                    help='number of layers in each column (default=10)')
parser.add_argument('-nintervals', type=int, default=60, metavar='N',
                    help='number of (equal) subintervals in computational domain (default=120)')
parser.add_argument('-R0', type=float, default=50.0e3, metavar='X',
                    help='half-width in m of ice sheet (default=50e3)')
parser.add_argument('-root', metavar='FILE', default='icegeom2d',
                    help='root of output file name (default=icegeom2d)')
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

# report on generated geometry and mesh
dxelem = 2.0 * args.L / args.nintervals
dyrefelem = args.Href / args.layers
PETSc.Sys.Print('generating 2D Halfar geometry on interval [%.2f,%.2f] km,'
                % (-args.L/1000.0,args.L/1000.0))
PETSc.Sys.Print('    with H0=%.2f m and R0=%.2f km, at t0=%.5f a,'
                % (args.H0,args.R0/1000.0,t0/secpera))
PETSc.Sys.Print('    as %d x %d node quadrilateral extruded mesh limited at Href=%.2f m,'
                % (args.nintervals,args.layers,args.Href))
PETSc.Sys.Print('    reference elements: dx=%.2f m, dy=%.2f m, ratio=%.5f'
                % (dxelem,dyrefelem,dyrefelem/dxelem))

base_mesh = IntervalMesh(args.nintervals, length_or_left=-args.L, right=args.L)
# note temporary layer_height:
mesh = ExtrudedMesh(base_mesh, layers=args.layers, layer_height=1.0/args.layers)

# deform y coordinate to match Halfar solution, but limited at Href
x,y = SpatialCoordinate(mesh)
Hinitial = halfar2d(x)
Hlimited = max_value(args.Href, Hinitial)
Vc = mesh.coordinates.function_space()
f = Function(Vc).interpolate(as_vector([x,Hlimited*y]))
mesh.coordinates.assign(f)

# space in which to solve Laplace equation (nabla^2 u = 0) with Dirichlet bcs
V = FunctionSpace(mesh, "CG", 1)

# simulate surface kinematical condition value for top; FIXME actual!
dt = secpera  # 1 year time steps
a = Constant(0.0)
smbref = conditional(Hinitial > args.Href, dt*a, dt*a - args.Href)  # FIXME
bcs = [DirichletBC(V, smbref, 'top'),
       DirichletBC(V, Constant(0.0), 'bottom')]

# Laplace equation weak form
u = Function(V, name='u')  # initialized to zero here
v = TestFunction(V)
F = dot(grad(u), grad(v)) * dx

# Solve system as though it is nonlinear:  F(u) = 0
solve(F == 0, u, bcs=bcs, options_prefix = 's')  # FIXME solver?

# visualize a function from this space with ParaView
PETSc.Sys.Print('writing ice geometry to %s ...' % (args.root+'.pvd'))
File(args.root+'.pvd').write(u)

