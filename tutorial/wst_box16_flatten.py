import numpy as np
import os

# This is my output directory of the coefficients (unflattened);
cache_dir = '/scratch/zxs/'

# Output path of the one-dimensional coefficients (flattened)
outpath='data_lo28_3'

# Total number of samples 
tot=1000

# The dimension of the box: 16*16*16;
M, N, O = 16, 16, 16

# Maximum J and L 
J=3
L=4

# The list of integral powers, which is the power applied after the modulus operation, i.e. Sum |f*g|^{integral_powers}
integral_powers = [0.5,1.0, 2.0,3,4]

# Number of intergral powers to be used
#q=len(integral_powers)
q=3 # use the first three q as example

# Take the logarithm of coefficients, average over L and integral_powers  
def s12_flatten(data):
    return np.average(np.log2(data[:,0:(L+1),0:q]),axis=1)

# Similar to s12, but handle the negative values by keeping the sign but taking log of the magnitude; I did not take the first value, integral_powers = 0.5 (data[1:]) and 1.0 (around 0 for mean-subtracted 21 cm field)
def s0_flatten(data):
    for ii in range(1,len(data)):
            if data[ii]>0:
                data[ii]=np.log2(np.abs(data[ii]))
            elif data[ii]<0:
                data[ii]=-np.log2(np.abs(data[ii]))
            else:
                data[ii]=0
    return data[2:] 

# Function to output the summary statistics (flattened one) for all samples
def summary(J):
    coef=[]
    for i in range(tot):
        data_s12=np.load(cache_dir+'s12_L_{:d}_J_{:d}_MNO_{}_num_{:d}.npy'.format(L,J,(M, N, O),i))
        s12=s12_flatten(data_s12)

        data_s0=np.load(cache_dir+'s0_L_{:d}_J_{:d}_MNO_{}_num_{:d}.npy'.format(L,J,(M, N, O),i))
        s0=np.array(s0_flatten(data_s0))

        # Concatenate s0, s1, and s2
        coef.append(np.concatenate([s0.flatten(),s12.flatten()]))
    coef=np.array(coef)
    return coef

# Final outputs by calling function summary(J)
stat=summary(J)
np.save('/scratch/zxs/{:s}/total_flatten_log_L_{:d}_J_{:d}_MNO_(66, 66, 66)_num_{:d}.npy'.format(outpath,L,J,tot),stat)
print (stat.shape)



