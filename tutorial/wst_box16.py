import numpy as np
import torch
import time
import os
from kymatio.torch import HarmonicScattering3D
from kymatio.scattering3d.backend.torch_backend \
    import compute_integrals
from kymatio.caching import get_cache_dir
import multiprocessing
import glob

# This is to choose the ID of the GPU on my machine, not tested on Infinity;
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# The dimension of the box: 16*16*16;
M, N, O = 16, 16, 16

# The actual maximum width of the dilated wavelet is 2*sigma*2^J=16, when the pixels in a box side is 16;
sigma = 1.0
J = 3

# Maximum angular frequency of harmonics: L=4;
L = 4

# The list of integral powers, which is the power applied after the modulus operation, i.e. Sum |f*g|^{integral_powers}
integral_powers = [0.5,1.0, 2.0,3,4]

# Up to 2nd-order coefficient;
max_order=2

# This is my output directory;
cache_dir = '/scratch/zxs/'

# Collect the files (simulated boxes) from the data directory;
data_dir = '/work/zxs/data_ic1_real'
matches = glob.glob(data_dir+'/*Idl*.npy')

# The directory to store my parameter files;
para_name='/scratch/zxs/mock/mocko_2_residual/'

# Initialize the scattering calculation;
scattering = HarmonicScattering3D(J=J, shape=(M, N, O),
                                  L=L, sigma_0=sigma,max_order=max_order,
                                  integral_powers=integral_powers)
# To cuda;
scattering.cuda()

# Initialize the lists for storing coefficients and the corresponding parameters (e.g. cosmological parameters);
coef=[]
para=[]

## Loop over the simulated boxes to calculate the coefficients;
# Batch size of the calculation at a time;
batchsize=8

for i in range(0,1000,batchsize):
    # image and parameter in batch;
    batch=[] 
    batch_p=[] 

    # Find the parameter-image pair files. My image file name example: Idlt-0.npy; corresponding parameter: Idlt-0-y.npy;
    for j in range(batchsize):  
        batch.append(np.load(files[i+j]))
        batch_p.append(np.load(para_name+'Idlt-'+files[i+j].split('/')[-1].split('-')[1].split('.')[0]+'-y.npy'))
    x=np.array(batch)
    y=np.array(batch_p)
    x=torch.from_numpy(x).float()
    x_gpu = x.cuda()

    # 1st and 2nd-order coefficients;
    order12_gpu = scattering(x_gpu) 
    order12_cpu = order12_gpu.cpu().numpy() 

    # Zeroth-order coefficients;
    order_0_gpu = compute_integrals(x,integral_powers)
    order_0=order_0_gpu.cpu().numpy()

    # Save the zeroth-order and 1st and 2nd-order coefficients;
    for k in range(batchsize):
        basename = 'L_{}_J_{}_MNO_{}_num_{}.npy'.format(L, J, (M, N, O),i+k)
        
        # 1st and 2nd;
        filename = os.path.join(cache_dir, 's12_' + basename)
        np.save(filename,order12_cpu[k])

        # Zeroth;
        filename = os.path.join(cache_dir, 's0_' + basename)
        np.save(filename,order_0[k])
        
        # Collect parameters;
        para.append(y[k])

# Save all parameters;
basename = 'L_{}_J_{}_MNO_{}_num_{}.npy'.format(L, J, (M, N, O),i+k+1)
filename = os.path.join(cache_dir, 'stot_y_' + basename)
para=np.array(para)
np.save(filename,para)

