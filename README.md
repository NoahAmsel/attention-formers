# attention-formers

Remember to update `python -m pip freeze > requirements.txt`

## refactor status
Still need to fix and test slurm.py
Ensure that the config is being assembled in the right function!

turn on batch size finder!

Now that we can run slurm from Python, we should continue with new_slurm.py

Look at Hydra's support for sweeps: https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/

```
python train.py fit --config configs/eg.yaml --model.nheads=16 --data.dim=16 --optimizer.lr=0.1 --trainer.fast_de
v_run=True
```

```
python slurm.py configs/slurm_config.yaml configs/eg.yaml --grid_path=configs/sweep.yaml > slurm/march24_cosine_sweep.slurm
```

TODO: try sweep with weight decay and with 1e-4
TODO: match "version" of csv logger with job_id. since right now things don't match up
TODO: off the shelf transformer with different numbers of heads and https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#transformerencoder