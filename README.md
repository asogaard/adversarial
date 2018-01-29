# adversarial

Tools for training and evaluating adversarially trained neural networks for
de-correlated jet tagging.



## Table of contents

- [Introduction](#introdution)
- [To-do](#to-do)
- [Quick start](#quick-start)
- [Environment](#environment)
  - [Anaconda](#anaconda)
  - [LCG](#lcg)
- [Supported platforms](#supported-platforms)
  - [Eddie3 compute cluster](#eddie3-compute-cluster)
- [Optimisation](#optimisation)
- [TensorBoard](#tensorboard)
- [Benchmarks](#benchmarks)
- [Known issues](#known-issues)
- [References](#references)



## Introduction

_To be added..._



## To-do

Below is a list of items which should ideally be addressed as part of the development and study of de-correlated tagger(s). Developers should feel free to assign themselves to any task they see fit.

**PUBNOTE**
- [ ] Get rel. 21 MC ntuples
- [ ] Complete code restructuring ([@asogaard])
- [ ] Validation tests
- [X] Setup optimisation framework ([@asogaard])
- [X] Implement re-weighting for flat prior ([@asogaard])
- [ ] Perform adversarial training with flatness re-weighting ([@asogaard])
- [ ] Determine scheme for feature selection
- [ ] Optimise classifier
- [ ] Determine optimisation metric for adversary
- [ ] Determine convergence condition for adversarial training
- [X] uBoost implementation ([@asogaard])
- [X] uBoost comparison study ([@asogaard])
- [ ] uBoost optimisation

**Data studies**
- [ ] Implementation in BoostedJetTaggers
- [ ] Get data/MC files from W/top-tagging CONF/paper
- [ ] Test on e.g. W/Z-peak in data



## Quick start

To get running on any [supported platform](#supported-platforms), do the following in a clean shell:

**Set up package**
```bash
$ git clone git@github.com:asogaard/adversarial.git
$ cd adversarial
$ source install.sh
$ source activate.sh
$ pip install -e .  # To be confirmed; may need to be done in each conda environment.
```
This installs the supported conda [environments](#environment) and activates
the one for CPU running.

**Stage some data**
```bash
$ source scripts/get_data.sh
```
If run elsewhere than lxplus, this will download a 1.4GB HDF5 data file.

**Test run**
```bash
$ python -m run.adversarial.train --help
$ python -m run.adversarial.train --train --tensorflow
```
This shows the supported arguments to the
[run/adversarial/train.py](run/adversarial/train.py) script, and starts training
using the TensorFlow backend. Several scripts — for training, testing, and
optimisation — are available in the [run/](run/) directory and can be run either
as modules or as scripts. Tab-completion will be enabled for all of them.



## Environment

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
```bash
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
[activate.sh](activate.sh) script.

```bash
$ source activate.sh cpu  # For running on CPU
$ source activate.sh gpu  # -              GPU
```
To deactivate the environment, do:
```bash
$ source activate.sh unset  # or
$ source deactivate
```


### LCG

On lxplus, the centrally provided SWAN LCG environment can used to set up most
of the required python packages, although generally older versions of
these. However, this is **not supported** and version conflicts are very likely
to occur. Should you wish to try it out anyway, it can be set up using:
```bash
$ source activate.sh lcg
```
with no installation required.



## Supported platforms

The code has been checked and found to work on the following operating systems: macOS 10.13 High
Sierra (local) and Scientific Linux 6/7 (lxplus/Eddie3), and CentOS 7 (lxplus7).

**Notice:** Although supported, it is not recommended to perform any substantial
 training on lxplus or on your personal computer, since they are (likely) not
 suited for the heavy computations required.


### Eddie3 compute cluster

Main wiki page describing the cluster is available
[here](https://www.wiki.ed.ac.uk/display/ResearchServices/Eddie). As Eddie3
provides compute nodes with up to 8 Nvidia Telsa K80 GPUs, this is a
recommended environment for training the networks.

#### Interactive sessions

To perform interactive test, log in to nodes with specific a parallel
environment and configuration. This is done like e.g.

```bash
$ qlogin -pe sharedmem 4 -l h_vmem=10G # CPU running
$ qlogin -pe gpu 4       -l h_vmem=10G # GPU
```

where the integer argument to the parallel environment argument (`-pe`) is the
number of CPUs/GPUs requested, and the value of the `h_vmem` is the requested
amount of memory per CPU. The `gpu` parallel environment provides one CPU for
each requested GPU.

#### Submitting jobs

To submit jobs to batch, do
```bash
$ ./submit.sh
```
which will submit data staging, training/evaluation, and finalisation jobs, in
that order. Use `TAB` to auto-complete and see available command-line arguments.



## Optimisation

To run the optimisation, simply do
```bash
$ source scripts/optimise.sh classifier
```
The argument refers to the named experiment in
[optimisation/experiments/](optimisation/experiments/). Several utility scripts
for the optimisation are available in
[optimisation/scripts/](optimisation/scripts/).



## TensorBoard

For convenience, the project natively supports TensorBoard for monitoring the training progress. To use TensorBoard, run using the `--tensorboard` flag, i.e.
```bash
$ python -m run.adversarial.train --train --tensorflow --tensorboard
```

The output TensorBoard is published to `http://localhost:6006` on the running server. If the code is run through ssh, it is still possible to access the results locally, by doing
```bash
$ ssh <user>@<host> -L 16006:127.0.0.1:6006
```

and navigating to `http://localhost:16006` on the local machine. The file outputs from running with TensorBoard callbacks are stored in the `logs/` directory of the project, and running TensorBoard manually is possible by doing
```bash
$ tensorboard --logdir logs/<timestamp>
```

Notice that TensorBoard requires using the TensorFlow backend. (_This might not be strictly true, but it's asserted nonetheless._)



## Benchmarks

The following shows the time per epoch, excluding overhead, used to train the
standalone classifier, running on 3.5M training samples (cross-val) per epoch
using the default network- and training configurations within the supported
conda environments. The devices used are Intel Xeon CPU E7-4820 v2 @ 2.00GHz
(CPU) and Nvidia Tesla K80 (GPU).

| **CPU** | **Theano**           | **Tensorflow**   |
|:-------:|:--------------------:|:---------------: |
| 1       | ca. 800 sec. / epoch | 245 sec. / epoch |
| 2       | _N/A_                | 210 sec. / epoch |
| 4       | _N/A_                | 185 sec. / epoch |
| **GPU** | **Theano**           | **Tensorflow**   |
| 1       |      52 sec. / epoch |  20 sec. / epoch |
| 2       | _N/A_                |  14 sec. / epoch |
| 4       | _N/A_                |  10 sec. / epoch |

Typically, the performance bottleneck is found to be data transfer, limiting GPU
utilisation to around 30-40%.



## Known issues

* Running in the CPU environment on macOS with the Theano backend does not
  support OpenMP due to use of clang compiler. See e.g. [here](https://groups.google.com/d/msg/theano-users/-oIdjtN-HmY/7jBixrHC6aAJ).



-----

## References

- G. Louppe, M. Kagan, and K. Cranmer, "Learning to Pivot with Adversarial Networks" [[arXiv:1611.01046](https://arxiv.org/abs/1611.01046)]
- C. Shimmin _et al._, "Decorrelated Jet Substructure Tagging using Adversarial Neural Networks". Phys. Rev. D 96, 074034 (2017) [[arXiv:1703.03507](https://arxiv.org/abs/1703.03507)]
- J. Stevens and M. Williams, "uBoost: A boosting method for producing uniform selection efficiencies from multivariate classifiers" [[arXiv:1305.7248](https://arxiv.org/abs/1305.7248)]



[@asogaard]: http://www.github.com/asogaard
