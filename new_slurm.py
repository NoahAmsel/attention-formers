from itertools import product
from typing import Optional

from jsonargparse import ArgumentParser
from omegaconf import OmegaConf as oc
from simple_slurm import Slurm
from simple_slurm import __path__ as simple_slurm_path


arguments_path = f"{simple_slurm_path[0]}/arguments.txt"
with open(arguments_path) as f:
    argument_names = f.read().replace(",", "\n").split("\n")


ExplicitSlurm_source = f"""
class ExplicitSlurm(Slurm):
    # Constructor should take all of the arugments named in arguments.txt and they should all be optional
    def __init__(
        self,
        {", ".join(f"{arg_name}:Optional[str]=None" for arg_name in argument_names)}
    ):
        # Slurm initializer is too dumb to ignore "None" so we must filter those out
        kwargs = dict({", ".join(f"{arg_name}={arg_name}" for arg_name in argument_names)})
        kwargs = {{k: v for k, v in kwargs.items() if v is not None}}
        super().__init__(**kwargs)
"""

exec(ExplicitSlurm_source)

# def generate(cfg):
#     print(cfg.cmd)
#     # slurm = Slurm(cfg.)


# def run(cfg):
#     pass


# if __name__ == "__main__":
#     generate_parser = ArgumentParser()
#     generate_parser.add_class_arguments(Slurm)

#     run_parser = ArgumentParser()
#     run_parser.add_argument("sweep_id", type=int)
#     run_parser.add_argument("job_id", type=int)

#     parser = ArgumentParser()
#     parser.add_argument("cmd", type=str)
#     parser.add_argument("--override_grid", enable_path=True, required=False)
#     subcommands = parser.add_subcommands()
#     subcommands.add_subcommand("generate", generate_parser)
#     subcommands.add_subcommand("run", run_parser)

#     # TODO: is there a way to avoid this boilerplate?
#     cfg = parser.parse_args()
#     if cfg.subcommand == "generate":
#         generate(cfg)
#     elif cfg.subcommand == "run":
#         run(cfg)

    # TODO! all arguments, including from all sweeps, should be validated
    # that means *not* being agnostic to the base command like I started coding it here


# idea: have the entire grid copied into the command that slurm runs
# in the form of a yaml list of config overrides
# that omegaconf / yaml file gets piped into a command
# that selects the appropriate config by task id, and then passes the corresponding args to the command string


def grid_to_list(grid):
    """Convert a grid to a list of configs."""
    return [dict(zip(grid.keys(), values)) for values in product(*grid.values())]


def read_grid(grid_path):
    return grid_to_list(oc.load(grid_path))


def override_dict2str(override_dict):
    return " ".join(f"--{k} {v}" for k, v in override_dict.items())

def launch(cmd_string, overrides, slurm_args):
    if overrides is None:
        overrides = [dict()]

    inside = "\n".join(f'\t{task_id})\n\t\tOVERRIDES="{override_dict2str(overrides[task_id-1])}"\n\t\t;;' for task_id in range(1, len(overrides)+1))
    overrides_cmd = f"""case $SLURM_ARRAY_TASK_ID in
{inside}

\t*)
\t\techo "ERROR! bad SLURM_ARRAY_TASK_ID"
\t\t;;
esac
"""

    full_command = f"""
singularity exec --nv --overlay /scratch/nia4240/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate attention
{cmd_string} ${{OVERRIDES}}
"
"""

    slurm_args["array"] = f"1-{len(overrides)}"
    slurm_args["a"] = f"1-{len(overrides)}"
    slurm = ExplicitSlurm(**slurm_args)
    slurm.add_cmd(overrides_cmd)
    slurm.add_cmd(full_command)
    print(slurm)

    # slurm.sbatch()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("cmd", type=str)
    parser.add_argument("--override_grid", enable_path=True, required=False)
    parser.add_class_arguments(ExplicitSlurm, nested_key="slurm", as_group=True)  # arguments to Slurm's __init__ aren't explicit, so these don't appear in help
    cfg = parser.parse_args()
    override_list = read_grid(cfg.override_grid) if cfg.override_grid else None
    launch(cfg.cmd, override_list, cfg.slurm)
