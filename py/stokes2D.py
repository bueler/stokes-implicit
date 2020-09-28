#!/usr/bin/env python3

from firedrake import *
import numpy as np

import argparse
parser = argparse.ArgumentParser(description=
'''
Generate 2D mesh from Halfar (1981) used extrusion of an equally-spaced
interval mesh.
''')
parser.add_argument('-H0', type=float, default=5000.0, metavar='X',
                    help='center height in m of ice sheet (default=5000)')
parser.add_argument('-Href', type=float, default=200.0, metavar='X',
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
args = parser.parse_args()

# physical constants
g = 9.81             # m s-2; constants in SI units
rho = 910.0          # kg
secpera = 31556926.0
n = 3.0
A = 1.0e-16/secpera
Gamma = 2.0 * A * (rho * g)**3.0 / 5.0   # see Bueler et al (2005)

# Halfar solution
# for t0, see equation (9) in Bueler et al (2005); result is 422.45 a
alpha = 1.0/11.0
t0 = (alpha/Gamma) * (7.0/4.0)**3.0 * (args.R0**4.0 / args.H0**7.0)
def halfar2d(x):
    absx = sqrt(pow(x,2))
    inpow = (n+1) / n
    outpow = n / (2*n+1)
    inside = max_value(0.0, 1.0 - pow(absx / args.R0,inpow))
    return args.H0 * pow(inside,outpow)

PETSc.Sys.Print('generating 2D Halfar geometry on interval [%.2f,%.2f] km,'
                % (-args.L/1000.0,args.L/1000.0))
PETSc.Sys.Print('    with H0=%.2f m and R0=%.2f km, at t0=%.5f a,'
                % (args.H0,args.R0/1000.0,t0/secpera))
PETSc.Sys.Print('    as %d x %d node quadrilateral extruded mesh,'
                % (args.nintervals,args.layers))
PETSc.Sys.Print('    limited at Href=%.2f m'
                % args.Href)

base_mesh = IntervalMesh(args.nintervals, length_or_left=-args.L, right=args.L)
# temporary layer_height
mesh = ExtrudedMesh(base_mesh, layers=args.layers, layer_height=1.0/args.layers)

# deform y coordiante to match Halfar solution
x,y = SpatialCoordinate(mesh)
Hlimited = max_value(args.Href, halfar2d(x))
Vc = mesh.coordinates.function_space()
f = Function(Vc).interpolate(as_vector([x,Hlimited*y]))
mesh.coordinates.assign(f)

# visualize a function from this space with ParaView
V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
PETSc.Sys.Print('writing ice geometry to %s ...' % (args.root+'.pvd'))
File(args.root+'.pvd').write(u)

