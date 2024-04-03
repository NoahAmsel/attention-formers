from itertools import accumulate, count, islice, product, takewhile
from math import comb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from models import OptimallyWeightedRandom
from task import NearestPointDatasetOrthogonal
from test import test_main
from test_cheating import slope
from theory_experiments.verify_joan import GegenbauerTransform


def num_harmonics(ambient_dim, level):
    dim = ambient_dim - 1
    if level == 0:
        return 1
    else:
        return comb(level + dim - 2, dim - 1) * (2*level + dim - 1) // level


def thresholds(ambient_dim, max_H):
    return list(takewhile(lambda x: x <= max_H, accumulate(num_harmonics(ambient_dim, level) for level in count())))


def analytic_error(dim, max_H):
    sgn = GegenbauerTransform(dim, np.sign, parity="odd")
    P_greater_than_deg = accumulate((-sgn.coeff(deg)**2 for deg in count()), initial=1)
    Hs = thresholds(dim, max_H)
    # TODO!
    print("TODO!")
    # actually, instead of just flatlining during the even degrees, maybe we should go straight on to the odd ones
    # or maybe each new head should simultaneously contribute to EVERY degree that isn't already filled up
    return Hs, list(islice(P_greater_than_deg, len(Hs)))
    # 0,0
    # N(d,0), eta_0^2
    # N(d,1), eta_0^2 + eta_1^2


def plot_H(curve, dim):
    s = slope(np.log(curve.index.to_numpy()), np.log(curve.to_numpy()))
    print(s)
    plt.loglog(curve)
    # plt.vlines(thresholds(dim, max(curve.index)), min(curve), max(curve))
    Hpoints, Epoints = analytic_error(dim, max(curve.index))
    plt.loglog(Hpoints, Epoints)
    plt.title(f"Measured MSE on q,k~unif, a=C^{{-1}}b\ndim={dim}\nslope={s:.2f}")
    plt.xlabel("H")
    plt.ylabel("MSE")


if __name__ == "__main__":
    config = dict(
        model=dict(
            model=dict(
                class_path="OptimallyWeightedRandom",
                init_args=dict(
                    num_gegenbauer_terms=30, scipy_solver=True,
                    # nheads=int(2**16), seed=0
                ),
            )
        ),
        data=dict(
            dataset_class=NearestPointDatasetOrthogonal,
            # dim=16,
            num_points=2,
            num_queries=1,
            batch_size=512,
            num_workers=3,
        ),
        trainer=dict(limit_test_batches=32),
    )
    # H = 2^16 took 7 mins with scipy_solve=True. (Memory problems with scipy_solve=False). beyond 2^16, who knows
    Hs = reversed([2**i for i in range(17)])
    dims = reversed([2**i for i in range(2, 7)])
    assert torch.cuda.is_available()
    results = []
    for seed, dim, H in tqdm(list(product(range(10), dims, Hs))):
        config["model"]["model"]["init_args"]["seed"] = seed
        config["data"]["dim"] = dim
        config["model"]["model"]["init_args"]["nheads"] = H
        squared_error = test_main(config)
        result = dict(seed=seed, dim=dim, H=H)
        print(result)
        results.append((dim, H, seed, squared_error))

    df = pd.DataFrame(results)
    filename = f"/home/nia4240/attention-formers/results/weighted_results_sweep_april3.csv"
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    df = pd.read_csv("results/weighted_results_sweep.csv")
    #### these files come from older sweeps and used larger numbers of heads
    df10 = pd.read_csv("results/weighted_results_d=10_H=65536.csv")
    df10["dim"] = 10
    df30 = pd.read_csv("results/weighted_results_d=30_H=65536.csv")
    df30["dim"] = 30
    df60 = pd.read_csv("results/weighted_results_d=60_H=16384.csv")
    df60["dim"] = 60
    df = pd.concat([df, df10, df30, df60], ignore_index=True)
    ####
    # curves = df.groupby(["dim", "H"])["Squared Error"].min()
    p = sns.lineplot(data=df, x="H", hue="dim", y="Squared Error", errorbar=("pi", 100))
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Measured MSE on q,k~unif, a=C^{{-1}}b")
    plt.xlabel("H")
    plt.ylabel("MSE")
    plt.savefig("results/optimal_weight.png", dpi=500)

    dfdfdf = df.loc[df.dim == 10]
    dfdfdf["?"] = np.log(df.dim)/np.log(df.H)
    # dfdfdf["??"] = 1/np.sqrt(df.H)
    plt.plot(dfdfdf.H, dfdfdf["Squared Error"], dfdfdf.H, dfdfdf["?"], marker="o")
    plt.xscale('log')
    plt.plot(dfdfdf.H, dfdfdf["Squared Error"] / dfdfdf["?"])

if __name__ == "__main__":
    df = pd.read_csv("results/weighted_results_sweep.csv")
    plot_dim = 64
    # should below be min?
    curve = df.groupby(["dim", "H"])["Squared Error"].mean().loc[plot_dim]
    plot_H(curve, plot_dim)


if __name__ == "__main__":
    plot_dim = 30
    df = pd.read_csv(f"results/weighted_results_d={plot_dim}_H=65536.csv")
    # should below be min? 
    curve = df.groupby("H")["Squared Error"].mean()
    plot_H(curve, plot_dim)
