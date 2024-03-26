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
                lr=metrics["lr-AdamW"].max(),
                rank=config.model.rank,
                nheads=config.model.nheads,
                dim=config.data.dim,
                num_points=config.data.num_points,
            ))
        except:
            pass
    return pd.DataFrame(run_data)


if __name__ == "__main__":
    df = experiment_analysis("/scratch/nia4240/attention-scratch/csv_logs/march24_cosine_sweep")
    df.to_csv("results/march20_big_sweep.csv")
    df_focus = df[(df.loss < 10) & (df.lr == 0.01)]
    p = sns.lineplot(data=df_focus, x="nheads", hue="dim", y="loss")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((0.1, 1.5))
    plt.ylabel("MSE")

    min_df = df.groupby(["nheads", "dim"])["loss"].min().reset_index()
    p = sns.lineplot(data=min_df, x="nheads", hue="dim", y="loss")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((0.1, 1.5))
    plt.title(f"Trained Custom Attention Layer, N=2")
    plt.xlabel("H")
    plt.ylabel("MSE")
    plt.savefig("results/march20_big_sweep", dpi=500)


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
                dim_feedforward=config.model.dim_feedforward
            ))
        except:
            pass
    return pd.DataFrame(run_data)


if __name__ == "__main__":
    df = experiment_analysis("/scratch/nia4240/attention-scratch/csv_logs/encoder_15_march25")
    df2 = experiment_analysis("/scratch/nia4240/attention-scratch/csv_logs/encoder_15_march25_part2")
    df = pd.concat([df, df2], ignore_index=True)
    df.to_csv("results/encoder_15_march25.csv")
    p = sns.lineplot(data=df, x="nheads", y="loss", hue="num_layers", errorbar=("pi", 100))
    plt.xscale('log')

    # min df
    min_df = df.groupby(["nheads", "num_layers"])["loss"].min().reset_index()
    p = sns.lineplot(data=min_df, x="nheads", y="loss", hue="num_layers")
    plt.xscale('log')
    plt.title("Standard Encoder: rank=dim/nheads\ndim=16, MLP width=2048, N=3")
    plt.savefig("results/encoder_15_march25.png", dpi=500)
