from itertools import product

import fire
from omegaconf import OmegaConf as oc
from simple_slurm import Slurm

from train import train


def read_configs(config_paths):
    return oc.merge(*[oc.load(config_path) for config_path in config_paths])


def grid_to_list(grid):
    """Convert a grid to a list of configs."""
    return [dict(zip(grid.keys(), values)) for values in product(*grid.values())]


def read_grid(grid_path):
    return grid_to_list(oc.load(grid_path))


def run_job(*config_paths, grid_path=None, sweep_id=0, job_id=0, debug=None):
    assert job_id > 0
    config = read_configs(config_paths)
    config.experiment_name = str(sweep_id)
    config.experiment_version = str(job_id)
    if debug is not None:
        config.debug = debug

    if grid_path is not None:
        overrides = read_grid(grid_path)[job_id - 1]
        for k, v in overrides.items():
            config[k] = v

    train(config)


def generate_slurm_file(*config_paths, grid_path=None):
    assert len(config_paths) >= 1
    config = read_configs(config_paths)

    if grid_path is None:
        num_jobs = 1
    else:
        sweep_list = read_grid(grid_path)
        num_jobs = len(sweep_list)

    slurm_file = Slurm(**config.slurm, array=f"1-{num_jobs}")

    command_string = f"""
singularity exec --nv --overlay {config.overlay}:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate {config.conda_env}
python -m fire slurm run_job {" ".join(config_paths)} --grid_path={grid_path} --sweep_id=$SLURM_ARRAY_JOB_ID --job_id=$SLURM_ARRAY_TASK_ID 
"
"""
    print(slurm_file)
    print(command_string)


if __name__ == '__main__':
    fire.Fire(generate_slurm_file)
