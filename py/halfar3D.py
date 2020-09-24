#!/usr/bin/env python3

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# TODO create .geo with mesh geometry from 2D and then 3D solutions

# HALFAR  Compute the similarity solution H(t,x,y) to the isothermal
# flat-bed SIA from Halfar (1983).  Time in seconds and coordinates x,y in
# meters.  Based on mccarthy/mfiles/halfar.m
# Constants H0 = 3600 m and R0 = 750 km are as in Test B in Bueler et al (2005).

debug = False
def figsave(name):
    print('saving %s ...' % name)
    if debug:
        plt.show()  # debug
    else:
        plt.savefig(name,bbox_inches='tight',transparent=True)

g = 9.81             # m s-2; constants in SI units
rho = 910.0          # kg
secpera = 31556926.0
n = 3.0
A = 1.0e-16/secpera
Gamma  = 2.0 * A * (rho * g)**3.0 / 5.0   # see Bueler et al (2005)

H0 = 3600.0
R0 = 750.0e3
alpha = 1.0/9.0
beta = 1.0/18.0
# for t0, see equation (9) in Bueler et al (2005); result is 422.45 a
t0 = (beta/Gamma) * (7.0/4.0)**3.0 * (R0**4.0 / H0**7.0)

t = 100.0 * secpera
L = 1000.0e3
xlist = np.linspace(-L,L,101)
x,y = np.meshgrid(xlist,xlist)

r = np.sqrt(x*x + y*y)
r = r / R0
t = t / t0
inside = np.maximum(0.0, 1.0 - (r / t**beta)**((n+1) / n))
H = H0 * inside**(n / (2*n+1)) / t**alpha

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x/1000.0,y/1000.0,H, cmap=cm.coolwarm)
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_zlabel('height (m)')

figsave('halfar.pdf')

