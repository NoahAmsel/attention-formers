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
        batch_size=64,
        num_test_batches=256,
        num_workers=0,  # this is necessary since we want to generate the dataset directly on the GPU, not in a CPU subprocess
    ))
    Hs = [2**i for i in range(17)]
    dims = [2**i for i in range(2, 7)]
    assert torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = []
    for dim, H, seed in tqdm(list(product(dims, Hs, range(10)))):
        with torch.no_grad():
            model = OptimallyWeightedRandom(dim, H, num_gegenbauer_terms=40, scipy_solver=True, seed=int(2e7 + seed), device=device)
            # since dataset is an iterable, must recreate it each inner
            # iteration to reset it
            data = dataset(oc.merge(config, {"dim": dim, "seed": int(1e7+seed)}), device=device)
            squared_error = model_mse(model, data, config.num_test_batches)
        # squared_error = float(test_model(model, oc.merge(config, {"seed": int(1e7+seed)})))
        results.append((dim, H, squared_error))

    df = pd.DataFrame(results, columns=["d", "H", "Squared Error"])
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
