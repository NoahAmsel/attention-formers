from pathlib import Path

import matplotlib.pyplot as plt
from numpy import log2
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
        additive_positional=config.model.get("additive_positional", False),
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


def fig1():
    df = pd.concat([
        experiment_analysis(f"/scratch/nia4240/attention-scratch/csv_logs/{exp_name}")
        for exp_name in ["fig1_4_25", "fig1_supp1"]
    ])
    variable_cols = list(df.nunique()[lambda x: x > 1].index)
    print(f"Variable columns:", *variable_cols)

    df.rename({"loss": "Mean Squared Error"}, axis=1, inplace=True)
    df["Rank"] = df["dim"] // df["nheads"]
    df["true num heads"] = df["nheads"] * df["width_multiplier"]
    df["Params per layer"] = (df["dim"] / df["nheads"]) * df["dim"] * df["true num heads"]
    df["Params per layer power"] = log2(df["Params per layer"]) / log2(df["dim"])
    df["Params per layer"] = df["Params per layer power"].apply(lambda x: f"$d^{{{x:.3g}}}$")

    fig, axs = plt.subplots(1, df["num_layers"].nunique(), figsize=(8, 5), sharey=True)
    for i, nlayers in enumerate(sorted(df["num_layers"].unique())):
        df_layer = df[df["num_layers"] == nlayers]
        df_grouped = df_layer.groupby(["Rank", "Params per layer", "additive_positional"])["Mean Squared Error"].min().reset_index()
        p = sns.lineplot(data=df_grouped, x="Rank", y="Mean Squared Error", style="Params per layer", hue="Params per layer", errorbar=("pi", 100), marker="o", hue_order=["$d^{2}$", "$d^{2.5}$", "$d^{3}$"], style_order=["$d^{2}$", "$d^{2.5}$", "$d^{3}$"], ax=axs[i])
        axs[i].set(xlabel='', title=f"Layers = {nlayers}")
        # axs[i].set(xscale='log')
        axs[i].set(yscale='log')
        if i > 0:
            # axs[i].set(ylabel=None)
            axs[i].legend([], [], frameon=False)
    plt.ylabel("MSE")
    fig.supxlabel("Rank")
    fig.tight_layout()
    plt.savefig(f"paper_experiments/imgs/fig1.png", dpi=500)
    return fig


def fig3():
    df = pd.concat([
        experiment_analysis(f"/scratch/nia4240/attention-scratch/csv_logs/{exp_name}")
        for exp_name in ["fig1_4_25", "fig1_supp1", "fig3_posn"]
    ])
    df.rename({"loss": "Mean Squared Error"}, axis=1, inplace=True)
    df["dim + posn"] = df["dim"] + df["positional_dim"]
    df["Rank"] = df["dim + posn"] // df["nheads"]
    df["true num heads"] = df["nheads"] * df["width_multiplier"]
    df["Params per layer"] = (df["dim"] / df["nheads"]) * df["dim"] * df["true num heads"]
    df["Params per layer power"] = log2(df["Params per layer"]) / log2(df["dim"])
    df["Params per layer"] = df["Params per layer power"].apply(lambda x: f"$d^{{{x:.3g}}}$")
    df["Positional Encoding"] = df.apply(
        lambda x: {
            (0, False): "None",
            (0, True): "Additive",
            (x["dim"], False): "Appended",
            (x["dim"], True): "Both",
        }[(x["positional_dim"], x["additive_positional"])],
        axis=1
    )
    df = df[df["num_layers"].isin([1, 3]) & (df["Params per layer"] == "$d^{2.5}$")]
    variable_cols = list(df.nunique()[lambda x: x > 1].index)
    print(f"Variable columns:", *variable_cols)

    fig, axs = plt.subplots(1, df["num_layers"].nunique(), figsize=(8, 5), sharey=True)
    for i, nlayers in enumerate(sorted(df["num_layers"].unique())):
        df_layer = df[df["num_layers"] == nlayers]
        df_grouped = df_layer.groupby(["Rank", "Params per layer", "Positional Encoding"])["Mean Squared Error"].min().reset_index()
        p = sns.lineplot(data=df_grouped, x="Rank", y="Mean Squared Error", style="Positional Encoding", hue="Positional Encoding", errorbar=("pi", 100), marker="o", hue_order=["None", "Additive", "Appended"], style_order=["None", "Additive", "Appended"], ax=axs[i])
        axs[i].set(xlabel='', title=f"Layers = {nlayers}")
        # axs[i].set(xscale='log')
        axs[i].set(yscale='log')
        if i != df["num_layers"].nunique() - 1:
            # axs[i].set(ylabel=None)
            axs[i].legend([], [], frameon=False)
    plt.ylabel("MSE")
    fig.supxlabel("Rank")
    fig.tight_layout()
    # plt.savefig(f"paper_experiments/imgs/fig3.png", dpi=500)
    return fig


if __name__ == "__main__":
    exp_name = "fig3_posn"
    log_path = f"/scratch/nia4240/attention-scratch/csv_logs/{exp_name}"
    df = experiment_analysis(log_path)
    variable_cols = list(df.nunique()[lambda x: x > 1].index)
    print(f"Variable columns:", *variable_cols)

    df.rename({"loss": "Mean Squared Error"}, axis=1, inplace=True)
    df["dim + posn"] = df["dim"] + df["positional_dim"]
    df["Rank"] = df["dim + posn"] // df["nheads"]
    df["true num heads"] = df["nheads"] * df["width_multiplier"]
    df["Params per layer"] = (df["dim"] / df["nheads"]) * df["dim"] * df["true num heads"]
    df["Params per layer power"] = log2(df["Params per layer"]) / log2(df["dim"])
    df["Params per layer"] = df["Params per layer power"].apply(lambda x: f"$d^{{{x:.3g}}}$")

    # first is x axis, then style, then hue
    df = df[df.positional_dim == 0]
    print(df["max_epoch"].describe())
    plot_vars = ["Rank", "Params per layer", "num_layers", "additive_positional"]
    # plot_vars = ["true num heads", "rank", "num_layers"]

    df_grouped = df.groupby(plot_vars)["Mean Squared Error"].min().reset_index()
    p = sns.lineplot(data=df_grouped, x=plot_vars[0], y="Mean Squared Error", style=plot_vars[1], hue=plot_vars[2] if len(plot_vars) > 2 else plot_vars[1], errorbar=("pi", 100), marker="o")
    plt.xlabel(plot_vars[0])
    plt.ylabel("MSE")
    plt.title(f"Standard Encoder: nheads = width_mult * dim / rank\ndim={get_nonunique(df, 'dim')}, MLP width={get_nonunique(df, 'dim_feedforward')}, positional dim={get_nonunique(df, 'positional_dim')}, data={get_nonunique(df, 'distribution')}")
    # plt.ylim([0, 1])
    plt.yscale('log')


if __name__ == "__main__":
    exp_name = "rms_hyperparam_sweep"
    log_path = f"/scratch/nia4240/attention-scratch/csv_logs/{exp_name}"
    df = experiment_analysis(log_path)
    df.sort_values(by="lr", inplace=True)
    df.groupby(["num_layers", "nheads", "bias", "lr"])[["loss", "max_epoch"]].mean()
    # turning bias off doesn't seem to consistently help
    # for 2 layers, lr = 0.05 is best and 0.01 is next
    # for 5 layers it's 0.001 followed by 0.01
