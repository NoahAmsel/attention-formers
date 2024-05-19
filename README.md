# attention-formers

To train a transformer model on the nearest neighbor task, run `python train.py fit <<OPTIONS>>`. See `python train.py fit --help` for explanations of all the options.
Options can also be specified by passing a config file: `python train.py fit --config <<CONFIG FILE>>`.
For an example, see `paper_experiments/fig1/fig1_config.yaml` or the main config file in any of the other folders in `paper_experiments`.

To conduct the experiments, run `sbatch` on each of the `.slurm` files in `paper_experiments`.

Figures from the paper are located in paper_experiments/imgs.
They were produed by running:
```
python analysis.py
python theory_experiments/optim_weight_linsolve.py
```

For environment details, see `requirements.txt`.
(Note to self: update with `python -m pip freeze > requirements.txt`.)
