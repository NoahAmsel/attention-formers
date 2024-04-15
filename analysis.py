from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import OmegaConf as oc
import pandas as pd
import seaborn as sns


def experiment_analysis(csv_logs_path):
    run_folders = Path(csv_logs_path).glob('**/version_*')
    run_data = []
    for run_folder in run_folders:
        try:
            metrics = pd.read_csv(Path(run_folder, "metrics.csv"))
            config = oc.load(Path(run_folder, "config.yaml"))
            run_data.append(dict(
                loss=metrics.train_loss.min(),
                lr=config.optimizer.lr,
                weight_decay=config.optimizer.weight_decay,
                nheads=config.model.nheads,
                dim=config.data.dim,
                num_points=config.data.num_points,
                num_layers=config.model.num_layers,
                positional_dim=config.model.positional_dim,
                dim_feedforward=config.model.dim_feedforward,
                width_multiplier=config.model.width_multiplier,
                # rank=config.model.rank,
            ))
        except:
            pass
    return pd.DataFrame(run_data)


def get_nonunique(df, col):
    vals = df[col].unique()
    assert len(vals) == 1
    return vals[0]


if __name__ == "__main__":
    exp_name = "widened_sweep"
    log_path = f"/scratch/nia4240/attention-scratch/csv_logs/{exp_name}"

    df = experiment_analysis(log_path)
    variable_cols = list(df.nunique()[lambda x: x > 1].index)
    print(f"Variable columns:", *variable_cols)

    df["rank"] = df["dim"] // df["nheads"]
    df["true num heads"] = df["nheads"] * df["width_multiplier"]
    # first is x axis, then style, then hue
    # df = df[(df.rank == 1) & (df.num_layers == 1)]
    plot_vars = ["true num heads", "rank", "num_layers"]

    df_grouped = df.groupby(plot_vars)["loss"].min().reset_index()
    p = sns.lineplot(data=df_grouped, x=plot_vars[0], y="loss", style=plot_vars[1], hue=plot_vars[2] if len(plot_vars) > 2 else plot_vars[1], errorbar=("pi", 100), marker="o")
    plt.xlabel(plot_vars[0])
    plt.ylabel("MSE")
    plt.title(f"Standard Encoder: rank=dim/nheads\ndim={get_nonunique(df, 'dim')}, MLP width={get_nonunique(df, 'dim_feedforward')}")
    plt.ylim([0, 1])
    plt.xscale('log')
    # plt.savefig(f"results/{exp_name}.png", dpi=500)
