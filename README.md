# adversarial

Main wiki page: https://www.wiki.ed.ac.uk/display/ResearchServices/Eddie


### Environment

This tool is used in the University of Edinburgh Eddie3 cluster, within the two following environments
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


### Submitting jobs

To submit jobs to batch, do
```
$ . scripts/submit.sh
```
which will submit data staging, training/evaluation, and finalisation jobs, in that order.


### Interactive sessions

To perform interactive test, login to nodes with specific a parallel environment and configuration. This is done like e.g.
```
$ qlogin -pe sharedmem 4    -l h_vmem=8G # CPU running
$ qlogin -pe gpu 4 -l gpu=4 -l h_vmem=8G # GPU 
```
where the integer argument to the parallel environment argument (`-pe`) is the number of CPU cores requested. The value of the `gpu` variable is the number of requeste GPUs, and the value of the `h_vmem` is the requested amount of memory per CPU.  
To quickly setup the interactive environment, do e.g.
```
$ source setup.sh          # Sets up CPU environment by default
$ source setup.sh gpu test # 'test' flag sets INPUTDIR and OUTPUTDIR environment variables
$ ./run.py -i $INPUTDIR -o $OUTPUTDIR --tensorflow --gpu
```
To unset the current environment, do
```
$ source setup.sh unset
```


### `matplotlib` fonts

Add custom `.ttf` fonts to
```
/exports/csce/eddie/ph/groups/PPE/asogaard/anaconda/adversarial-{cpu,gpu}/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/
```
and remember to clear fonts cache
```
$ rm ~/.cache/matplotlib/fontList.cache
```
