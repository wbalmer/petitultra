# low level imports
import os
import sys
os.environ["OMP_NUM_THREADS"] = "1" # stop e.g. numpy from doing own parallel with mpi
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # keep hdf5 file locking up across processes 
sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL) # for rockfish dlopen error

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


import scipy.stats
import numpy as np
from numpy import sin, pi
import ultranest
import ultranest.stepsampler
from ultranest.plot import cornerplot


def sine_model2(t, B, A1, P1, t1, A2, P2, t2):
    return A1 * sin((t / P1 + t1) * 2 * pi) + A2 * sin((t / P2 + t2) * 2 * pi) + B

np.random.seed(42)

n_data = 50

# time of observations
t = np.random.uniform(0, 5, size=n_data)
# measurement values
yerr = 1.0
y = np.random.normal(sine_model2(t, B=1.0, A1=4.2, P1=3, t1=0, A2=1.2, P2=1.2, t2=1.2), yerr)


parameters2 = ['B', 'A1', 'P1', 't1', 'A2', 'P2', 't2']

def prior_transform2(cube):
    # the argument, cube, consists of values from 0 to 1
    # we have to convert them to physical scales

    params = cube.copy()
    # let background level go from -10 to +10
    params[0] = cube[0] * 20 - 10
    # let amplitude go from 0.1 to 100
    params[1] = 10**(cube[1] * 3 - 1)
    # let period go from 0.3 to 30
    params[2] = 10**(cube[2] * 2)
    # let time go from 0 to 1
    params[3] = cube[3]

    # let amplitude go from 0.01 to 100
    params[4] = 10**(cube[4] * 3 - 1)
    # let period go from 0.3 to 30
    params[5] = 10**(cube[5] * 2)
    # let time go from 0 to 1
    params[6] = cube[6]
    return params

def log_likelihood2(params):
    # unpack the current parameters:
    B, A1, P1, t1, A2, P2, t2 = params

    # avoid unnecessary multiple solutions:
    #    force ordering by period from large to small
    if P1 < P2:
        # instead of returning a very low number:
        ## return -1e300
        # which would give a likelihood plateau causing some loss of live points
        # we give a slope towards the "good" parameter space:
        return -1e300 * abs(P1 - P2)

    # compute for each x point, where it should lie in y
    y_model = sine_model2(t, B=B, A1=A1, P1=P1, t1=t1, A2=A2, P2=P2, t2=t2)
    # compute likelihood
    loglike = -0.5 * (((y_model - y) / yerr)**2).sum()

    return loglike

sampler2 = ultranest.ReactiveNestedSampler(
    parameters2,
    log_likelihood2,
    prior_transform2,
    wrapped_params=[False, False, False, True, False, False, True],
)

# have to choose the number of steps the slice sampler should take
# after first results, this should be increased and checked for consistency.

nsteps = 2 * len(parameters2)

# create step sampler:
sampler2.stepsampler = ultranest.stepsampler.SliceSampler(
    nsteps=nsteps,
    generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
    # adaptive_nsteps=False,
    # max_nsteps=400
)

# run again:
result2 = sampler2.run(min_num_live_points=400)
sampler2.print_results()

comm.Barrier()

if rank == 0:
    import matplotlib.pyplot as plt
    cornerplot(result2)
    plt.show()

