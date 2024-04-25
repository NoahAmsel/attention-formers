from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import OmegaConf as oc
import pandas as pd
import seaborn as sns


def convergence_curve(run_folder, burnin=0):
    metrics = pd.read_csv(Path(run_folder, "metrics.csv"))[["epoch", "train_loss"]].dropna()
    plt.figure()
    plt.plot(metrics[int(burnin*len(metrics)):].set_index("epoch"))
    desc = extract_experiment_description(run_folder)
    plt.title(f"{run_folder.split('/')[-2:]}\nnum_layers={desc['num_layers']}, nheads={desc['nheads']}, lr={desc['lr']}")
    # plt.yscale("log")


def all_convergence_curves(exp_name):
    csv_logs_path = f"/scratch/nia4240/attention-scratch/csv_logs/{exp_name}"
    run_folders = Path(csv_logs_path).glob('**/version_*')
    for run_folder in run_folders:
        convergence_curve(str(run_folder), .1)


def distribution_name(dataset_class):
    return {
        "task.NearestPointDataset": "unif",
        "task.NearestPointDatasetOrthogonal": "ortho"
    }[dataset_class]


def extract_experiment_description(run_folder):
    metrics = pd.read_csv(Path(run_folder, "metrics.csv"))
    config = oc.load(Path(run_folder, "config.yaml"))
    return dict(
        loss=metrics.train_loss.min(),
        max_epoch=metrics.epoch.max(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        nheads=config.model.nheads,
        dim=config.data.dim,
        num_points=config.data.num_points,
        num_layers=config.model.num_layers,
        positional_dim=config.model.positional_dim,
        dim_feedforward=config.model.dim_feedforward,
        width_multiplier=config.model.width_multiplier,
        distribution=distribution_name(config.data.dataset_class),
        bias=config.model.bias,
        # rank=config.model.rank,
    )


def experiment_analysis(csv_logs_path):
    run_folders = Path(csv_logs_path).glob('**/version_*')
    run_data = []
    for run_folder in run_folders:
        try:
            run_data.append(extract_experiment_description(run_folder))
        except:
            print("Failed to read: ", run_folder)
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
    df["Num parameters"] = df["width_multiplier"] * df["dim"]
    # first is x axis, then style, then hue
    df = df[df.positional_dim == 0]
    plot_vars = ["rank", "width_multiplier", "num_layers"]
    # plot_vars = ["true num heads", "rank", "num_layers"]

    df_grouped = df.groupby(plot_vars)["loss"].min().reset_index()
    p = sns.lineplot(data=df_grouped, x=plot_vars[0], y="loss", style=plot_vars[1], hue=plot_vars[2] if len(plot_vars) > 2 else plot_vars[1], errorbar=("pi", 100), marker="o")
    plt.xlabel(plot_vars[0])
    plt.ylabel("MSE")
    plt.title(f"Standard Encoder: nheads = width_mult * dim / rank\ndim={get_nonunique(df, 'dim')}, MLP width={get_nonunique(df, 'dim_feedforward')}, positional dim={get_nonunique(df, 'positional_dim')}, data={get_nonunique(df, 'distribution')}")
    plt.ylim([0, .8])
    plt.xscale('log')
    # plt.savefig(f"results/{exp_name}.png", dpi=500)

    fig, axs = plt.subplots(1,3, figsize=(8, 5), sharey=True)
    for i, nlayers in enumerate(sorted(df["num_layers"].unique())):
        df_layer = df[df["num_layers"] == nlayers]
        df_grouped = df_layer.groupby(["rank", "Num parameters"])["loss"].min().reset_index()
        p = sns.lineplot(data=df_grouped, x="rank", y="loss", style="Num parameters", hue="Num parameters", errorbar=("pi", 100), marker="o", ax=axs[i])
        axs[i].set(ylim=[0, .8], xlabel='', title=f"Layers = {nlayers}")
        # axs[i].set(xscale='log')
        if i < 2:
            # axs[i].set(ylabel=None)
            axs[i].legend([], [], frameon=False)
    fig.suptitle(f"Standard Encoder: num params = rank * num heads\ndim={get_nonunique(df, 'dim')}, MLP width={get_nonunique(df, 'dim_feedforward')}, positional dim={get_nonunique(df, 'positional_dim')}, data={get_nonunique(df, 'distribution')}")
    plt.ylabel("MSE")
    fig.supxlabel("rank")
    fig.tight_layout()



if __name__ == "__main__":
    exp_name = "rms_hyperparam_sweep"
    log_path = f"/scratch/nia4240/attention-scratch/csv_logs/{exp_name}"
    df = experiment_analysis(log_path)
    df.sort_values(by="lr", inplace=True)
    df.groupby(["num_layers", "nheads", "bias", "lr"])[["loss", "max_epoch"]].mean()
    # turning bias off doesn't seem to consistently help
    # for 2 layers, lr = 0.05 is best and 0.01 is next
    # for 5 layers it's 0.001 followed by 0.01
