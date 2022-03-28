# 21cmDELFI

**21cmDELFI**. This package is for the density estimation likelihood inference (DELFI) with 21 cm statistics like the power spectrum (21cmDELFI-PS) ([Zhao et al. 2022a](https://arxiv.com/)). The main module is heavily borrowed from the [pydelfi](https://github.com/justinalsing/pydelfi) package, mainly described in [Alsing, Charnock, Feeney and Wandelt 2019](https://arxiv.org/abs/1903.00007), with minor changes in `delfi.py`. The validations of the posteriors are partly borrowed from [galpo](https://github.com/smucesh/galpro/), which is based on [S. Mucesh et al. 2021](https://academic.oup.com/mnras/article/502/2/2770/6105325). The other validation tool is based on [Diana Harrison et al. 2015](https://academic.oup.com/mnras/article/451/3/2610/1186451). If you use this code, please consider citing these papers.

**Installation:**

The code is tested with python3 (3.7) and the [tensorflow](https://www.tensorflow.org) (1.14) in a separate conda environment. Other dependencies:<br>
[getdist](http://getdist.readthedocs.io/en/latest/)<br>
[emcee](http://dfm.io/emcee/current/) (>=3.0.2)<br>
[tqdm](https://github.com/tqdm/tqdm)<br>
[mpi4py](https://mpi4py.readthedocs.io/en/stable/) (if MPI is required)<br>

To install the dependencies and this package, you can first run
```
pip install tensorflow==1.14
```
then clone the repo and run `python setup.py install` from the top-level directory.

For the active learning with the 21 cm power spectrum, you should have the following two dependencies:<br>
[21cmFAST](https://github.com/andreimesinger/21cmFAST) <br>
[21cmSense](https://github.com/steven-murray/21cmSense) <br>

**Documentation and tutorials:** 

If you want to implement your own 21cm signal, please check the scripts in the `tutural` directory, where you can replace the data with your own.<br> 

For validation of both marginal and joint posteriors, you can check the `Diagnostics.ipynb` under the `tutural` directory.

You may also want to compress the 21 cm images into different low-dimensional summaries. For example, in [this repository](https://github.com/Xiaosheng-Zhao/DELFI-3DCNN), we present the code used to train a 3DCNN [(Zhao et al. 2022b)](https://arxiv.org/abs/2105.03344) as the data compressor.

The main documentation of [pydelfi](https://github.com/justinalsing/pydelfi) can be found **[here](https://pydelfi.readthedocs.io/en/latest/)**.

