import logging, sys, os
import numpy as np
import matplotlib.pyplot as plt
import pydelfi.priors as priors
import pydelfi.ndes as ndes
import pydelfi.delfi as delfi
import tensorflow as tf
import pickle
from getdist import plots, MCSamples
import getdist

logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)
import py21cmfast as p21c
from py21cmfast import plotting
from py21cmfast import cache_tools
from tpower import get_power
import random
from mpi4py import MPI

restore=False
#one set of parameters could correspond to multiple realizations of noise
n_noise=1

def simulator(theta,seed,simulator_args,batch=1,n_noise=n_noise):
    p21c.config['direc'] = "./data/cache"
    nc = 66
    redshift = [7.668188,8.003182,8.357909,8.716638,9.108561,9.524624,9.966855,10.437500,10.939049,11.474274]
    coeval8 = p21c.run_coeval(
    redshift = redshift,
    write=False,
    user_params = {"HII_DIM": 66, "BOX_LEN": 100},
    cosmo_params = p21c.CosmoParams(SIGMA_8=0.815,hlittle=0.678,OMm=0.308,OMb=0.048425,POWER_INDEX=0.968),
    astro_params = p21c.AstroParams({"HII_EFF_FACTOR":pow(10,theta[1]),"ION_Tvir_MIN":theta[0]}),
    random_seed=seed)
    ps = np.concatenate((get_power(coeval8[0],nc,n_noise,redshift[0]),get_power(coeval8[1],nc,n_noise,redshift[1]),get_power(coeval8[2],nc,n_noise,redshift[2]),get_power(coeval8[3],nc,n_noise,redshift[3]),get_power(coeval8[4],nc,n_noise,redshift[4]),get_power(coeval8[5],nc,n_noise,redshift[5]),get_power(coeval8[6],nc,n_noise,redshift[6]),get_power(coeval8[7],nc,n_noise,redshift[7]),get_power(coeval8[8],nc,n_noise,redshift[8]),get_power(coeval8[9],nc,n_noise,redshift[9])),axis=1)
    return ps

# Simulator arguments
simulator_args = None
def compressor(d, compressor_args):
    return d
compressor_args = None

# Define the priors parameters
lower = np.array([4,1])
upper = np.array([6,2.398])
prior = priors.Uniform(lower, upper)
theta_fiducial = np.array([5.47712125,2.30103])

seed = 4101110
nout = 80

data=np.load('/scratch/zxs/delfi_fast/data/21cmdelfi_mock/back_1003/bright_hera.npy')
compressed_data = compressor(data, compressor_args)

# Create an ensemble of NDEs
NDEs = [ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=2, n_data=nout, n_hiddens=[50,50], n_mades=5, act_fun=tf.tanh, index=0),
            ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=2, n_data=nout, n_hiddens=[50,50], n_mades=5, act_fun=tf.tanh, index=1),
            ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=2, n_data=nout, n_hiddens=[50,50], n_mades=5, act_fun=tf.tanh, index=2),
            ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=2, n_data=nout, n_hiddens=[50,50], n_mades=5, act_fun=tf.tanh, index=3)]

# Initiate the MPI setting
world_comm=MPI.COMM_WORLD
print (world_comm.Get_rank(),flush=True)
# Create the DELFI object
DelfiEnsemble = delfi.Delfi(compressed_data, prior, NDEs, Finv=None, theta_fiducial=theta_fiducial,
                       param_limits = [lower, upper],
                       param_names =['\mathrm{log_{10}\,T_{vir}}', '\mathrm{log_{10}\,\zeta}'],
                       results_dir = "./data/results",
                       restore = restore,
                       n_procs = 10,
                       comm=world_comm,
                       red_op=MPI.SUM,
                       n_noise=n_noise,
                       rank=world_comm.Get_rank(),
                       input_normalization=None)

# Initial samples, batch size for population samples, number of populations
n_initial = 500
n_batch = 500
n_populations = 60

# Do the SNL training
DelfiEnsemble.sequential_training(simulator, compressor, n_initial, n_batch, n_populations, patience=20, batch_size = 100,plot = False,save_intermediate_posteriors=True)
x0 = DelfiEnsemble.posterior_samples[np.random.choice(np.arange(len(DelfiEnsemble.posterior_samples)), p=DelfiEnsemble.posterior_weights.astype(np.float32)/sum(DelfiEnsemble.posterior_weights), replace=False, size=DelfiEnsemble.nwalkers),:]
posterior_samples = DelfiEnsemble.emcee_sample(x0=x0,main_chain=2000)

with open('data/posterior_samples/po_mock_active.pkl', 'wb') as f:
    pickle.dump(posterior_samples,f)
