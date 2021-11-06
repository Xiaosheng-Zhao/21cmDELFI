# pydelfi

**21cmDELFI**. This package is heavily borrowed from the [pydelfi](https://github.com/justinalsing/pydelfi) package with minor changes. The implemented methods are described in detail in [Xiaosheng Zhao et al. 2021a](https://arxiv.com/) and  [Xiaosheng Zhao et al. 2021](https://arxiv.org/abs/2105.03344). The [pydelfi](https://github.com/justinalsing/pydelfi) package are mainly described in [Alsing, Charnock, Feeney and Wandelt 2019](https://arxiv.org/abs/1903.00007), and are based closely on [Papamakarios, Sterratt and Murray 2018](https://arxiv.org/pdf/1805.07226.pdf), [Lueckmann et al 2018](https://arxiv.org/abs/1805.09294) and [Alsing, Wandelt and Feeney, 2018](https://academic.oup.com/mnras/article-abstract/477/3/2874/4956055?redirectedFrom=fulltext). Please cite these papers if you use this code!

**Installation:**

The code is in python3. There is a Tensorflow 1 (most stable) and Tensorflow 2 version that can be installed as follows:<br>

**Tensorflow 1 (stable)**

This can be found on the master branch and has the following dependencies:<br>
[tensorflow](https://www.tensorflow.org) (<=1.15) <br> 
[getdist](http://getdist.readthedocs.io/en/latest/)<br>
[emcee](http://dfm.io/emcee/current/) (>=3.0.2)<br>
[tqdm](https://github.com/tqdm/tqdm)<br>
[mpi4py](https://mpi4py.readthedocs.io/en/stable/) (if MPI is required)<br>

You can install the requirements and this package with,
```
pip install git+https://github.com/justinalsing/pydelfi.git
```
or alternatively, pip install the requirements and then clone the repo and run `python setup.py install`


**Documentation and tutorials:** 

If you want to implement your own 21cm signal, please check the scripts in the "tutural" directory, where you can replace the data with your own.<br> 

The main documentation from [pydelfi](https://github.com/justinalsing/pydelfi) can be found **[here](https://pydelfi.readthedocs.io/en/latest/)**.

