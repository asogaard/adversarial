# adversarial

Tools for training and evaluating adversarially trained neural networks for
de-correlated jet tagging.



## Contents

TBA


## Quick start

To get running on any [supported platform](#supported-platforms), do the following in a clean shell:

**Set up package**
```
$ git clone git@github.com:asogaard/adversarial.git
$ cd adversarial
$ source install.sh
$ source setup.sh
```
This installs the supported conda [environments](#environments) and activates
the one for CPU running.

**Stage some data**
```
$ source scripts/get_data.sh
```
If run elsewhere than lxplus, this will download a 1.4GB HDF5 data file.

**Test run**
```
$ ./run.py --help
$ ./run.py --train --tensorflow
```
This shows the supported arguments to the [run.py](run.py) script, and starts
training using the TensorFlow backend. Tab-completion is enabled for
[run.py](run.py).



## <a name="environments">Environment</a>

The preferred method to set up the python environment required to run the code
is to use [Anaconda](https://conda.io/docs/), which ensures that all clones of
the library are run in exactly the same environment. Alternatively, on lxplus,
the centrally provided LCG environment can be used. However, this is **not
supported** and will require modifying the code to accommodate package version
differences.


### Anaconda

To use the custom, supported anaconda environment, simply run the
[install.sh](install.sh) script. It (optionally) **installs conda** and
**creates the supported environments**.

#### Install conda

If `conda` is not installed already, it is **done automatically** during the
installation. Alternatively, you can do it manually by logging on to your
preferred platform, e.g. lxplus, and doing the following:
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

#### Create the conda environment(s)

The [install.sh](install.sh) script also creates two separate conda
environments, `adversarial-cpu` and `adversarial-gpu`, for running the code on
CPU and GPU, respectively, using the `.yml` environment snapshots in
[envs/](envs/). This ensures that all users are running in the exact same
enviroment.

#### Activate the environment(s)

Everytime you are starting a new shell, before running the adversarial neural
network code, you should activate the installed environment by using the
[setup.sh](setup.sh) script.

```
$ source setup.sh cpu  # For running on CPU
$ source setup.sh gpu  # -              GPU
```
To deactivate the environment, do:
```
$ source setup.sh unset  # or
$ source deactivate
```


### LCG

On lxplus, the centrally provided SWAN LCG environment can used to set up most
of the required python packages, although generally older versions of
these. However, this is **not supported** and version conflicts are very likely
to occur. Should you wish to try it out anyway, it can be set up using:
```
$ source setup.sh lcg
```
with no installation required.



## <a name="supported-platforms">Supported platforms</a> 

The code has been checked and found to work on the following operating systems: macOS 10.13 High
Sierra (local), CentOS 6/7 (lxplus/lxplus7), and Scientific Linux 7 (Eddie3).

**Notice:** Although supported, it is not recommended to perform any substantial
 training on lxplus or on your personal computer, since they are (likely) not
 suited for the heavy computations required.


### University of Edinburgh Eddie3 compute cluster

Main wiki page describing the cluster is available
[here](https://www.wiki.ed.ac.uk/display/ResearchServices/Eddie). As Eddie3
provides compute nodes with up to 8 Nvidia Telsa K80 GPU's, this is a
recommended environment for training the networks.

#### Interactive sessions

To perform interactive test, log in to nodes with specific a parallel
environment and configuration. This is done like e.g.

```
$ qlogin -pe sharedmem 4 -l h_vmem=10G # CPU running
$ qlogin -pe gpu 4       -l h_vmem=10G # GPU 
```

where the integer argument to the parallel environment argument (`-pe`) is the
number of CPUs/GPUs requested, and the value of the `h_vmem` is the requested
amount of memory per CPU. The `gpu` parallel environment provides one CPU for
each requested GPU.

#### Submitting jobs

To submit jobs to batch, do
```
$ ./submit.sh
```
which will submit data staging, training/evaluation, and finalisation jobs, in
that order. Use `TAB` to auto-complete and see available command-line arguments.
