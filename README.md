# adversarial

## Contents

TBA


## Running on lxplus

**Notice:** Although supported, it is not recommended to perform any substantial
 training on lxplus, since the nodes are not suited for the heavy computations
 required.


### Environment

The preferred method to set up the python environment required to run the code
is to use [Anaconda](https://conda.io/docs/), which ensures that all clones of
the library are run in exactly the same environment. Alternatively, the
centrally provided LCG environment can be used.

#### Anaconda

To instead use the custom, supported anaconda environment, do the
[following](https://conda.io/docs/user-guide/tasks/manage-environments.html#building-identical-conda-environments):

- **Install miniconda**

If `conda` is not installed already, it is **done automatically during
installation**. Alternatively, you can do it manually by logging on to your
preferred cluster, e.g. lxplus, and doing the following
```
$ wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
$ bash Miniconda2-latest-Linux-x86_64.sh
$ # Follow the screen prompts
$ # ...
$ rm Miniconda2-latest-Linux-x86_64.sh
``` 
This installs the conda package manager, allowing us to setup a custom
environment, common for all installations of this package. Please ensure that
your system variable `PATH` points to the location of your conda installation.

- **Create the conda environment(s)**

```
$ # After having cloned the 'adversarial' package
$ cd adversarial
$ source install.sh
```
This creates two separate conda environments, `adversarial-{cpu,gpu}`, for
running the code on CPU and GPU, respectively, using the `.yml` environment
snapshots in [envs/](envs/), which ensures that all users are running in the
exact same enviroment

- **Activate the environment(s)**

Everytime your starting a new shell, you should activate the installed
environment by using the [setup.sh](setup.sh) script.
```
$ source setup.sh cpu

or

$ source setup.sh gpu

Do your thing...

$ source setup.sh unset
```

#### LCG

On lxplus, the centrally provided SWAN LCG environment can used to set up the
required python packages. However, this is not supported and version conflicts
are very likely to occur. This can be set up using
```
$ source setup.sh lcg
```

### Quick start

To get running on lxplus, do
```
$ cd my/work/directory
$ git clone git@github.com:asogaard/adversarial.git
$ cd adversarial
$ source install.sh
$ source setup.sh
$ ./run.py --help
$ ./run.py --train --tensorflow --devices 8 --input /eos/atlas/user/a/asogaard/adversarial/data/
```



## Running on Eddie3 computing cluster

Main wiki page: https://www.wiki.ed.ac.uk/display/ResearchServices/Eddie


### Environment

This tool is used on the University of Edinburgh Eddie3 cluster, within the two following environments
```
$ module load anaconda cuda/8.0.61 root/6.06.02
$ conda config --add envs_dirs /exports/csce/eddie/ph/groups/PPE/asogaard/anaconda
$ conda create  -n adversarial-{cpu,gpu} python=2.7.13-0 numpy=1.13.1 scipy=0.19.1 matplotlib=2.0.2 pip=9.0.1
$ conda install -n adversarial-{cpu,gpu} -c anaconda    tensorflow{,-gpu}=1.2.1
$ conda install -n adversarial-{cpu,gpu} -c conda-forge keras=2.0.6
$ conda install -n adversarial-{cpu,gpu} mkl-service
$ conda install -n adversarial-{cpu,gpu} graphviz
$ conda install -n adversarial-gpu pygpu
$ source activate adversarial-{cpu,gpu}
$ pip install root_numpy
$ pip install hep_ml
$ pip install pydot-ng
$ pip install graphviz
$ pip install psutil
$ source deactivate
```


### Interactive sessions

To perform interactive test, login to nodes with specific a parallel environment and configuration. This is done like e.g.
```
$ qlogin -pe sharedmem 4 -l h_vmem=8G # CPU running
$ qlogin -pe gpu 4       -l h_vmem=8G # GPU 
```
where the integer argument to the parallel environment argument (`-pe`) is the number of CPU cores requested. The value of the `gpu` variable is the number of requeste GPUs, and the value of the `h_vmem` is the requested amount of memory per CPU.  
To quickly setup the interactive environment, do e.g.
```
$ source setup.sh          # Sets up CPU environment by default
$ source setup.sh gpu test # 'test' flag sets INPUTDIR and OUTPUTDIR environment variables
$ ./run.py -i $INPUTDIR -o $OUTPUTDIR --tensorflow --gpu --devices 4
```
Tab-completion is enabled for `run.py`.
To unset the current environment, do
```
$ source setup.sh unset
```


### Submitting jobs

To submit jobs to batch, do
```
$ ./submit.sh
```
which will submit data staging, training/evaluation, and finalisation jobs, in
that order. Used `TAB` to auto-complete and see available command-line arguments.
