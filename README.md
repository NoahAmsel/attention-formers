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

### TODO
- CSV log folder should be experiment_name/sweep_id/job_id so that we can run multiple sweeps as part of the same experiment. if we see that some died or we need to add a few extra
    - this also protects us from accidentally overwriting csv logs if we use the same experiment name twice
- understand if/why it started using DPP. make sure this is being done properly. read tutorial, make sure the seeds are being set correctly
- turn on batch size finder!
- try sweep with weight decay and with 1e-4
- should log sweep id and job id
- should log git revision
- off the shelf transformer with different numbers of heads layers
    - each encoder block has layer norm. how does this affect our problem?
    - https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#transformerencoder
- use wandb?
