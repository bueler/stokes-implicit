# module for ice constants

# earth
secpera = 31556926.0
g = 9.81             # m s-2; constants in SI units

# ice
rho = 910.0          # kg
n = 3.0              # warning: this value is hardwired in some places below
A3 = 1.0e-16/secpera

# computed constants
B3 = A3**(-1.0/3.0)                       # Pa s(1/3);  ice hardness
Gamma = 2.0 * A3 * (rho * g)**3.0 / 5.0   # see Bueler et al (2005)

