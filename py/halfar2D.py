#!/usr/bin/env python3

# Generate a meshable domain in the plane (halfar2D.geo) from the 2D similarity
# solution H(t0,x) to the isothermal flat-bed SIA from Halfar (1981).
# Defaults: H0 = 5000 m and R0 = 50 km gives 20-to-1 aspect.

import argparse
parser = argparse.ArgumentParser(description=
'''Generate .geo geometry-description file, suitable for meshing by Gmsh, from
a 2D similarity solution H(t0,x) to the isothermal flat-bed SIA (Halfar, 1981).
The geometry is the reference domain for a coupled Stokes and surface
kinematical equation implicit time step.  Defaults: H0 = 5000 m and R0 = 50 km
gives 20-to-1 aspect for fluid.
''')
parser.add_argument('-H0', type=float, default=5000.0, metavar='X',
                    help='center height in m of ice sheet (default=5000)')
parser.add_argument('-Href', type=float, default=200.0, metavar='X',
                    help='minimum thickness in m of reference domain (default=200)')
parser.add_argument('-image', action='store_true', default=False,
                    help='write additional .pdf image')
parser.add_argument('-L', type=float, default=60.0e3, metavar='X',
                    help='half-width in m of computational domain (default=60e3)')
parser.add_argument('-nintervals', type=int, default=120, metavar='N',
                    help='number of (equal) subintervals in computational domain (default=120)')
parser.add_argument('-R0', type=float, default=50.0e3, metavar='X',
                    help='half-width in m of ice sheet (default=50e3)')
parser.add_argument('-root', metavar='FILE', default='halfar2D',
                    help='root of output file name (default=halfar2D)')
args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt

def figsave(name,debug=False):
    print('saving image %s ...' % name)
    if debug:
        plt.show()  # debug
    else:
        plt.savefig(name,bbox_inches='tight',transparent=True)

# physical constants
g = 9.81             # m s-2; constants in SI units
rho = 910.0          # kg
secpera = 31556926.0
n = 3.0
A = 1.0e-16/secpera
Gamma = 2.0 * A * (rho * g)**3.0 / 5.0   # see Bueler et al (2005)

alpha = 1.0/11.0
# for t0, see equation (9) in Bueler et al (2005); result is 422.45 a
t0 = (alpha/Gamma) * (7.0/4.0)**3.0 * (args.R0**4.0 / args.H0**7.0)
print('generating 2D Halfar geometry with H0=%.2f m, R0=%.2f km, and t0=%.5f a'
      % (args.H0,args.R0/1000.0,t0/secpera))

x = np.linspace(-args.L,args.L,args.nintervals+1)
dx = 2.0 * args.L / args.nintervals
inside = np.maximum(0.0, 1.0 - (abs(x) / args.R0)**((n+1) / n))
H = args.H0 * inside**(n / (2*n+1))

if args.image:
    fig = plt.figure()
    plt.plot(x/1000.0,H,'k')
    plt.xlabel('x (km)')
    plt.ylabel('height (m)')
    #plt.axis('equal')
    figsave(args.root + '.pdf')

from datetime import datetime
import sys, platform, subprocess

filename = args.root + '.geo'
print('writing domain geometry to file %s ...' % filename)
#args = processopts()
# HEADER
commandline = " ".join(sys.argv[:])  # for save in comment in generated .geo
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
geo = open(filename, 'w')
# header which records creation info
geo.write('// geometry-description file created %s by %s using command\n//   %s\n\n'
          % (now,platform.node(),commandline) )
# CHARACTERISTIC LENGTHS
geo.write('cl = %f;\n' % dx)
geo.write('Href = %f;\n' % args.Href)

# points in counter-clockwise order from lower left
M = 1   # M is the object index in the .geo
# run across the base from left to right
for k in range(args.nintervals+1):
    geo.write('Point(%d) = {%f,0.0,0.0,cl};\n' % (M,x[k]))
    M = M+1
# FIXME
for k in range(args.nintervals+1):
    j = args.nintervals - k
    if H[j] < args.Href:
        geo.write('Point(%d) = {%f,Href,0.0,cl};\n' % (M,x[j]))
    else:
        geo.write('Point(%d) = {%f,%f,0.0,cl};\n' % (M,x[j],H[j]))
    M = M+1
Mlastpoint = M-1

# lines along boundary
Mfirstline = M
for k in range(2*args.nintervals+1):
    ln = 2*args.nintervals+2 + k+1
    geo.write('Line(%d) = {%d,%d};\n' % (M,k+1,k+2))
    M = M+1
geo.write('Line(%d) = {%d,%d};\n' % (M,k+2,1))
M = M+1
# line loop
Mlineloop = M
geo.write('Line Loop(%d) = {%d' % (M,Mfirstline))
for k in range(2*args.nintervals+1):
    geo.write(',%d' % (Mfirstline+k+1))
geo.write('};\n')
M = M+1
# plane surface
Mplanesurface = M
geo.write('Plane Surface(%d) = {%d};\n' % (M,M-1))
M = M+1
# physical line:
Mphysicalline = M
geo.write('Physical Line(%d) = {%d' % (M,Mfirstline))
for k in range(2*args.nintervals+1):
    geo.write(',%d' % (Mfirstline+k+1))
geo.write('};\n')
M = M+1
# physical surface
geo.write('Physical Surface(%d) = {%d};\n\n' % (M,Mplanesurface))
geo.close()


