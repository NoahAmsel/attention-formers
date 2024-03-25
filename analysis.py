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
    df = experiment_analysis("csv_logs/march24_cosine_sweep")
    df = df[df.loss < 10]
    df = df[df.lr == 0.001]
    p = sns.lineplot(data=df, x="nheads", hue="dim", y="loss", errorbar=("pi", 100))
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((0.1, 1.5))
    plt.title(f"Measured MSE on q,k~unif, a=C^{{-1}}b")
    plt.xlabel("H")
    plt.ylabel("MSE")
