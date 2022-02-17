# 21cmDELFI

**21cmDELFI**. This package is for the simulation-based inference with 21cm statistics. It is heavily borrowed from the [pydelfi](https://github.com/justinalsing/pydelfi) package with minor changes. The implemented methods are described in detail in and  [Xiaosheng Zhao et al. 2022a](https://arxiv.org/abs/2105.03344) [Xiaosheng Zhao et al. 2021b](https://arxiv.com/). The [pydelfi](https://github.com/justinalsing/pydelfi) package are mainly described in [Alsing, Charnock, Feeney and Wandelt 2019](https://arxiv.org/abs/1903.00007), and are based closely on [Papamakarios, Sterratt and Murray 2018](https://arxiv.org/pdf/1805.07226.pdf), [Lueckmann et al 2018](https://arxiv.org/abs/1805.09294) and [Alsing, Wandelt and Feeney, 2018](https://academic.oup.com/mnras/article-abstract/477/3/2874/4956055?redirectedFrom=fulltext). Please cite these papers if you use this code!

**Installation:**

The code is in python3 and can be installed as follows:<br>

This code  has the following dependencies:<br>
[tensorflow](https://www.tensorflow.org) (<=1.15) <br> 
[getdist](http://getdist.readthedocs.io/en/latest/)<br>
[emcee](http://dfm.io/emcee/current/) (>=3.0.2)<br>
[tqdm](https://github.com/tqdm/tqdm)<br>
[mpi4py](https://mpi4py.readthedocs.io/en/stable/) (if MPI is required)<br>

For the active learning with the 21 cm power spectrum, we should have the following two dependencies:<br>
[21cmFAST](https://github.com/andreimesinger/21cmFAST) <br>
[21cmSense](https://github.com/steven-murray/21cmSense) <br>

You can install the requirements and this package with,
```
pip install git+https://github.com/Xiaosheng-Zhao/21cmDELFI.git
```
or alternatively, pip install the requirements and then clone the repo and run `python setup.py install`


**Documentation and tutorials:** 

If you want to implement your own 21cm signal, please check the scripts in the "tutural" directory, where you can replace the data with your own.<br> 

In [this repository](https://github.com/Xiaosheng-Zhao/DELFI-3DCNN), we present the code used to train a 3DCNN [(Xiaosheng Zhao et al. 2022a)](https://arxiv.org/abs/2105.03344) as the data compressor.

The main documentation from [pydelfi](https://github.com/justinalsing/pydelfi) can be found **[here](https://pydelfi.readthedocs.io/en/latest/)**.

