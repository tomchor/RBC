"""
Dedalus script for 3D Rayleigh-Benard convection.

This script uses parity-bases in the x and y directions to mimick no-slip,
insulating sidewalls.  The equations are scaled in units of the thermal
diffusion time (Pe = 1).

This script should be ran in parallel, and would be most efficient using a
2D process mesh.  It uses the built-in analysis framework to save 2D data slices
in HDF5 files.  The `merge.py` script in this folder can be used to merge
distributed analysis sets from parallel runs, and the `plot_2d_series.py` script
can be used to plot the slices.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 merge.py snapshots
    $ mpiexec -n 4 python3 plot_2d_series.py snapshots/*.h5

The simulation should take roughly 400 process-minutes to run, but will
automatically stop after an hour.

"""

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
scale = 3
ratio = 5

lz = 1
nz = 32*scale
nx, nz = (ratio*nz, nz)
Lx, Lz = (ratio*lz, lz)
print(f"Ratio: {ratio}\nΔz = {lz/nz}\nTotal size = {nz*nx} points\n")

epsilon = 0.8
Pr = 1.0
Ra = 1e9

dt_0 = 1e-10/scale

# Create bases and domain
start_init_time = time.time()
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'], time='t')
#problem.meta['p','b','w','bz','wz']['x']['parity'] = 1
#problem.meta['u','uz']['x']['parity'] = -1
problem.parameters['P'] = Pr
problem.parameters['R'] = Ra
problem.parameters['F'] = F = 1

problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(b) - (dx(dx(b)) + dz(bz))                   = - u*dx(b) - w*bz")
problem.add_equation("dt(u) - P*(dx(dx(u)) + dz(uz)) + dx(p)         = - u*dx(u) - w*uz")
problem.add_equation("dt(w) - P*(dx(dx(w)) + dz(wz)) + dz(p) - R*P*b = - u*dx(w) - w*wz")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")

problem.add_bc("left(b) = -left(F*z)")
problem.add_bc("left(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(b) = -right(F*z)")
problem.add_bc("right(u) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("integ_z(p) = 0", condition="(nx == 0)")

# Build solver
#solver = problem.build_solver(de.timesteppers.MCNAB2)
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions
z = domain.grid(1)
b = solver.state['b']
bz = solver.state['bz']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=23)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-2 * noise * (zt - z) * (z - zb)
b['g'] = -F*(z - pert)
b.differentiate('z', out=bz)

# Integration parameters
solver.stop_sim_time = 200
solver.stop_wall_time = 60 * 60. * 24 * 4 # maximum number of seconds
solver.stop_iteration = np.inf

# Analysis
snap = solver.evaluator.add_file_handler('snapshots_ra', sim_dt=0.2, max_writes=10)
snap.add_task("p", scales=1, name='p')
snap.add_task("b", scales=1, name='b')
snap.add_task("u", scales=1, name='u')
snap.add_task("w", scales=1, name='w')

# CFL
#CFL = flow_tools.CFL(solver, initial_dt=dt_0, cadence=5, safety=1.5,
#                     max_change=1.5, min_change=0.1, max_dt=1e-2)
CFL = flow_tools.CFL(solver, initial_dt=dt_0, max_dt=1e-2, cadence=5, safety=1.5, max_change=1.5)
CFL.add_velocities(('2*u', '2*w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("u*u + w*w", name='KE')

# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max KE = %f' %flow.max('KE'))
        if np.isnan(flow.max('KE')):
            break
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))
    logger.info('Final KE: %f' %(flow.max('KE')))


