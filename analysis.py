from pathlib import Path

from matplotlib import ticker
import matplotlib.pyplot as plt
from numpy import log2, pi
from omegaconf import OmegaConf as oc
import pandas as pd
import seaborn as sns

from train import PerfectEncoderRegression


def convergence_curve(run_folder, burnin=0):
    metrics = pd.read_csv(Path(run_folder, "metrics.csv"))[["epoch", "train_loss"]].dropna()
    plt.figure()
    plt.plot(metrics[int(burnin*len(metrics)):].set_index("epoch"))
    desc = extract_experiment_description(run_folder)
    plt.title(f"{run_folder.split('/')[-2:]}\nnum_layers={desc['num_layers']}, nheads={desc['nheads']}, lr={desc['lr']}")
    # plt.yscale("log")


def all_convergence_curves(exp_folder):
    run_folders = Path(exp_folder).glob('**/version_*')
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
        nheads=config.model.nheads,
        num_layers=config.model.num_layers,
        positional_dim=config.model.positional_dim,
        additive_positional=config.model.get("additive_positional", False),
        dim_feedforward=config.model.dim_feedforward,
        width_multiplier=config.model.width_multiplier,
        bias=config.model.bias,
        batch_size=config.data.batch_size,
        dim=config.data.dim,
        num_points=config.data.num_points,
        distribution=distribution_name(config.data.dataset_class),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        run_folder=run_folder
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


def fig1(csv_logs):
    df = pd.concat([
        experiment_analysis(f"{csv_logs}/{exp_name}")
        for exp_name in ["fig1_4_25", "fig1_supp1"]
    ])
    ##
    df = df[df['lr'] == 0.01]
    ##
    variable_cols = list(df.nunique()[lambda x: x > 1].index)
    print(f"Variable columns:", *variable_cols)

    df.rename({"loss": "Mean Squared Error"}, axis=1, inplace=True)
    df["Rank"] = df["dim"] // df["nheads"]
    df["true num heads"] = df["nheads"] * df["width_multiplier"]
    df["Params per layer"] = (df["dim"] / df["nheads"]) * df["dim"] * df["true num heads"]
    df["Params per layer power"] = log2(df["Params per layer"]) / log2(df["dim"])
    df["Params per layer"] = df["Params per layer power"].apply(lambda x: f"$d^{{{x:.3g}}}$")

    fig, axs = plt.subplots(1, df["num_layers"].nunique(), figsize=(8, 3.5), sharey=True)
    for i, nlayers in enumerate(sorted(df["num_layers"].unique())):
        df_layer = df[df["num_layers"] == nlayers]
        p = sns.lineplot(
            data=df_layer,
            x="Rank",
            y="Mean Squared Error",
            style="Params per layer",
            hue="Params per layer",
            estimator="min",
            errorbar=None,
            # errorbar=("pi", 100),
            marker="o",
            hue_order=["$d^{2}$", "$d^{2.5}$", "$d^{3}$"],
            style_order=["$d^{2}$", "$d^{2.5}$", "$d^{3}$"],
            ax=axs[i],
        )
        axs[i].set(xlabel="", yscale='log', title=f"Layers = {nlayers}")
        axs[i].set(xscale='log')
        axs[i].set_xticks([1,2,4,8,16,32,64])
        axs[i].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        axs[i].get_xaxis().set_tick_params("minor", size=0, width=0)
        if i > 0:
            axs[i].legend([], [], frameon=False)
    plt.ylabel("MSE")
    fig.supxlabel("Rank")
    fig.tight_layout()
    plt.savefig("paper_experiments/imgs/fig1.png", dpi=500)
    return fig


def fig_perfect(csv_logs):
    run_folders = Path(f"{csv_logs}/fig1_4_25").glob('**/version_*')
    run_data = []
    for run_folder in run_folders:
        config = oc.load(Path(run_folder, "config.yaml"))
        if (config.model.nheads == 1) and (config.model.num_layers == 1):
            # TODO: load best epoch, not last
            model = PerfectEncoderRegression.load_from_checkpoint(f"{run_folder}/checkpoints/last.ckpt")
            for ix in range(config.model.width_multiplier):
                QK, VO = model.get_QK_VO(ix)
                QK_angle, QK_norm = model.compare_to_identity(QK)
                VO_angle, VO_norm = model.compare_to_identity(VO)
                run_data.append(
                    dict(
                        width_multiplier=config.model.width_multiplier,
                        QK_angle=float(QK_angle),
                        QK_norm=float(QK_norm),
                        VO_angle=float(VO_angle),
                        VO_norm=float(VO_norm),
                    )
                )

    df = pd.DataFrame(run_data)
    df["Angle btwn.  $KQ^\\top$ and $I$"] = df["QK_angle"] * 180 / pi
    df.rename({"width_multiplier": "Num heads", "QK_norm": "$\|KQ^\\top\|_F$"}, axis=1, inplace=True)

    fig, axs = plt.subplots(1, 2, figsize=(8, 2.4))
    sns.boxplot(data=df, x="Num heads", y="Angle btwn.  $KQ^\\top$ and $I$", ax=axs[0])
    axs[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}°"))
    # axs[0].set_ylim([0, 180])
    # axs[0].set_yticks([0, 30, 60, 90, 120, 150, 180])
    axs[0].set_ylim([160, 180])
    sns.boxplot(data=df, x="Num heads", y="$\|KQ^\\top\|_F$", ax=axs[1]) #  estimator="median", errorbar=("pi", 50), 
    axs[1].set_ylim([0, 3000])
    plt.tight_layout()
    plt.savefig("paper_experiments/imgs/fig_perfect.png", dpi=500)
    return fig


def fig3(csv_logs):
    df = pd.concat([
        experiment_analysis(f"{csv_logs}/{exp_name}")
        for exp_name in ["fig1_4_25", "fig3_posn"]
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
            (x["dim"], False): "Concatenated",
            (x["dim"], True): "Both",
        }[(x["positional_dim"], x["additive_positional"])],
        axis=1
    )
    df = df[df["num_layers"].isin([1, 3]) & (df["Params per layer"] == "$d^{2.5}$") & (df["Rank"] <= 64)]
    variable_cols = list(df.nunique()[lambda x: x > 1].index)
    print(f"Variable columns:", *variable_cols)

    fig, axs = plt.subplots(1, df["num_layers"].nunique(), figsize=(7 * .8, 3.5 * .8), sharey=True)
    for i, nlayers in enumerate(sorted(df["num_layers"].unique())):
        df_layer = df[df["num_layers"] == nlayers]
        p = sns.lineplot(
            data=df_layer,
            x="Rank",
            y="Mean Squared Error",
            style="Positional Encoding",
            hue="Positional Encoding",
            estimator="min",
            # errorbar=None,
            errorbar=("pi", 100),
            marker="o",
            hue_order=["None", "Additive", "Concatenated"],
            style_order=["None", "Additive", "Concatenated"],
            ax=axs[i],
        )
        axs[i].set(xlabel="", yscale='log', title=f"Layers = {nlayers}")
        axs[i].set(xscale='log')
        axs[i].set_xticks([1,2,4,8,16,32,64])
        axs[i].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        axs[i].get_xaxis().set_tick_params("minor", size=0, width=0)
        if i > 0:
            axs[i].legend([], [], frameon=False)
    plt.ylabel("MSE")
    fig.supxlabel("Rank")
    fig.tight_layout()
    plt.savefig("paper_experiments/imgs/fig3.png", dpi=500)
    return fig


def figN(csv_logs):
    df = experiment_analysis(f"{csv_logs}/figN")
    variable_cols = list(df.nunique()[lambda x: x > 1].index)
    print(f"Variable columns:", *variable_cols)

    df.rename({"loss": "Mean Squared Error", "num_points": "Num Points"}, axis=1, inplace=True)
    df["Rank"] = df["dim"] // df["nheads"]
    df["true num heads"] = df["nheads"] * df["width_multiplier"]
    df["Params per layer"] = (df["dim"] / df["nheads"]) * df["dim"] * df["true num heads"]
    df["Params per layer power"] = log2(df["Params per layer"]) / log2(df["dim"])
    df["Params per layer"] = df["Params per layer power"].apply(lambda x: f"$d^{{{x:.3g}}}$")

    df["Num Points"] = df["Num Points"].apply(str)  # this is to make the hue prettier
    fig = plt.figure(figsize=(5.5 * .8, 4.0 * .8))
    p = sns.lineplot(
        data=df,
        x="Rank",
        y="Mean Squared Error",
        style="Num Points",
        hue="Num Points",
        hue_order=["256", "64", "16", "4"],
        estimator="min",
        errorbar=("pi", 100),
        marker="o",
    )
    plt.xscale("log")
    p.set_xticks([1,2,4,8,16,32,64])
    p.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    p.get_xaxis().set_tick_params("minor", size=0, width=0)
    fig.tight_layout()
    plt.savefig("paper_experiments/imgs/figN.png", dpi=500)
    return fig


if __name__ == "__main__":
    plt.rcParams["font.family"] = "serif"
    csv_logs_folder = "/scratch/nia4240/attention-scratch/csv_logs/"
    fig1(csv_logs_folder)
    fig_perfect(csv_logs_folder)
    fig3(csv_logs_folder)
    figN(csv_logs_folder)
