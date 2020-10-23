# module for ice constants

# earth
secpera = 31556926.0
g = 9.81              # m s-2; constants in SI units

# ice
rho = 910.0           # density; kg
n = 3.0               # warning: this value is needed for some parts
An = 1.0e-16/secpera  # softness; EISMINT 1 value

# computed constants
Bn = An**(-1.0/n)     # Pa s(1/n);  ice hardness
Gamma = 2.0 * An * (rho * g)**n / (n+2.0)

