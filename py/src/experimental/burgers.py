# modification of:  https://www.firedrakeproject.org/demos/burgers.py.html

from firedrake import *
n = 30
mesh = UnitSquareMesh(n, n)

V = VectorFunctionSpace(mesh, "CG", 2)
u_ = Function(V, name="Velocity")
u = Function(V, name="VelocityNext")
v = TestFunction(V)

x = SpatialCoordinate(mesh)
#ic = project(as_vector([sin(pi*x[0]), 0]), V)
ic = project(as_vector([sin(pi*x[0]), sin(2.0*pi*x[1])]), V)

# Set u to the initial condition, also used as our starting guess.
u_.assign(ic)
u.assign(ic)

# diffusion rate small
#nu = 0.0001
nu = 0.01

# The timestep is set to produce an advective Courant number of
# around 1. Since we are employing backward Euler, this is stricter than
# is required for stability, but ensures good temporal resolution.
timestep = 1.0/n

# Here we finally get to define the residual of the equation. In the advection
# term we need to contract the test function :math:`v` with 
# :math:`(u\cdot\nabla)u`, which is the derivative of the velocity in the
# direction :math:`u`. This directional derivative can be written as
# ``dot(u,nabla_grad(u))`` since ``nabla_grad(u)[i,j]``:math:`=\partial_i u_j`.
# Note once again that for a nonlinear problem, there are no trial functions in
# the formulation. These will be created automatically when the residual
# is differentiated by the nonlinear solver::
F = (inner((u - u_)/timestep, v)
     + inner(dot(u,nabla_grad(u)), v) + nu*inner(grad(u), grad(v)))*dx

u.rename('velocity')
outfile = File("burgers.pvd")
outfile.write(u)

t = 0.0
end = 0.5
while (t <= end):
    print('  t = %.5f' % t)
    solve(F == 0, u)
    outfile.write(u)
    u_.assign(u)
    t += timestep

