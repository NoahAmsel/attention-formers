# attention-formers

Remember to update `python -m pip freeze > requirements.txt`

## refactor status
Still need to fix and test slurm.py
Ensure that the config is being assembled in the right function!

Now that we can run slurm from Python, we should continue with new_slurm.py

Look at Hydra's support for sweeps: https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

```
python train.py fit --config configs/eg.yaml --model.nheads=16 --data.dim=16 --optimizer.lr=0.1 --trainer.fast_de
v_run=True
```

```
python slurm.py configs/slurm_config.yaml configs/eg.yaml --grid_path=configs/sweep.yaml > slurm/march24_cosine_sweep.slurm
```

On hyperparam selection, David sent this: https://arxiv.org/abs/2309.14322

### node / gpu / worker selection
- the following experiments were all with dim=64
- one GPU (gv006) with 15 cpus
    - for biggest item in the sweep, 3 workers, a single epoch of 100 batches of size 1024. 100% gpu utilization
        - 3 workers took 5 mins 56 sec
        - same thing with 15 workers took 5 mins 29 sec. 100% gpu utilization
    - now for the smallest item in the sweep. in all cases, utilization is 20%. time to do 4 epochs
        - 3 workers took 13 sec
        - 7 workers took 8.4 sec. utilizaion 20%
        - 15 workers took 7 sec. utilizaiton 20%
    - I don't trust the timings with low utilization. so i made a setting that was just big enough so that utilization is 100%. under this setting. this setting has 32 points. time to do 4 epochs:
        - 3 workers took 23 sec
        - 15 workers took 24 sec
    - but really the worker time depends only on number of points. same setting but raising number of points to 256
        - 3 workers took 4 min 10 sec to do only 3 epochs
        - 15 workesr took about the same


- running `lscpu | grep -E '^Thread|^Core|^Socket|^CPU\('` yielded
```
CPU(s):                             48
Thread(s) per core:                 1
Core(s) per socket:                 24
Socket(s):                          2
```
socket * cores per socket = # cpus

### TODO
- check if adding more workers helps speed or not...
    - for the smallest job in our sweep, it did help up to 3 jobs when we have 4 cpus. still should see if it's worth requesting 16 or whatever
    - for the biggest config in our sweep, it didn't help to have 3 jobs vs. 1.
- maybe think it started using data parallelism because my slurm jobs got assigned the more nodes. should I now reduce the number of nodes so it WON"T use DPP? (can uncomment out single node strategy)
- understand if/why it started using DPP. make sure this is being done properly. read tutorial, make sure the seeds are being set correctly
    - was it because my slurm job got assigned more nodes or something? should I reduce cpus-per-task to stop this?
    - was it from downgrading pytorch-lightning or using lightning-bolts?
    - was it from requesting too much memory??
- check to see how Greene nodes are structured
- turn on batch size finder!
- try sweep with weight decay and with 1e-4
- should log sweep id and job id
- should log git revision
- off the shelf transformer with different numbers of heads layers
    - each encoder block has layer norm. how does this affect our problem?
    - https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#transformerencoder
- use wandb?

### Resolved (I think)
- CSV log folder should be experiment_name/sweep_id/job_id so that we can run multiple sweeps as part of the same experiment. if we see that some died or we need to add a few extra
    - this also protects us from accidentally overwriting csv logs if we use the same experiment name twice
- double check that the dataset isn't being "reset" at the start of each epoch. we want to double check that we're never reusing examples
