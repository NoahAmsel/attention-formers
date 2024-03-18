from itertools import product

import matplotlib.pyplot as plt
from omegaconf import OmegaConf as oc
import pandas as pd
import torch
from tqdm import tqdm

from models import OptimallyWeightedRandom
from task import dataset
from test_cheating import model_mse, slope


if __name__ == "__main__":
    config = oc.create(dict(
        num_points=2,
        num_queries=1,
        task="ortho",
        scale_batch=False,
        # I set this to avoid memory overflow when num heads > 2^15,
        # but it makes inefficient use of GPU. 
        # TODO: dynamically adjust batch size to fit model size
        batch_size=512,
        num_workers=0,  # this is necessary since we want to generate the dataset directly on the GPU, not in a CPU subprocess
    ))
    config.num_test_batches = 16_384 // config.batch_size
    Hs = reversed([2**i for i in range(15)])
    dims = reversed([2**i for i in range(2, 7)])
    assert torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = []
    for seed, dim, H in tqdm(list(product(range(10), dims, Hs))):
        with torch.no_grad():
            model = OptimallyWeightedRandom(dim, H, num_gegenbauer_terms=40, scipy_solver=False, seed=int(2e7 + seed), device=device)
            # since dataset is an iterable, must recreate it each inner
            # iteration to reset it
            data = dataset(oc.merge(config, {"dim": dim, "seed": int(1e7+seed)}), device=device)
            squared_error = model_mse(model, data, config.num_test_batches)
        # squared_error = float(test_model(model, oc.merge(config, {"seed": int(1e7+seed)})))
        results.append((dim, H, seed, squared_error))
        with open("/home/nia4240/attention-formers/running_sweep_results.csv", "a") as f:
            print(dim, H, seed, squared_error, sep=",", file=f)

    df = pd.DataFrame(results, columns=["dim", "H", "seed", "Squared Error"])
    filename = f"/home/nia4240/attention-formers/weighted_results_sweep.csv"
    df.to_csv(filename, index=False)
    df = pd.read_csv(filename)
    curve = df.groupby("H")["Squared Error"].mean()

    s = slope(torch.log(Hs), torch.log(torch.tensor(curve.to_numpy())))
    print(s)
    plt.semilogy(Hs, curve)
    plt.title(f"Measured MSE on q,k~unif, a=C^{{-1}}b\ndim={config.dim}\nslope={s:.2f}")
    plt.xlabel("H")
    plt.ylabel("MSE")

    # import seaborn as sns
    # df["1/MSE"] = 1/df['Squared Error']
    # p = sns.lineplot(df, x='H', y='1/MSE')
    # p.set(xscale='log')
    # p.set(yscale='log')
