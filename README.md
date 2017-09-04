# adversarial

Main wiki page: https://www.wiki.ed.ac.uk/display/ResearchServices/Eddie

### Environment

This tool is used in the University of Edinburgh Eddie3 cluster, within the following environment
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
$ source deactivate
```

### Submitting jobs

To submit jobs to batch, do
```
$ . scripts/submit.sh
```
which will submit data staging, training/evaluation, and finalisation jobs, in that order.
