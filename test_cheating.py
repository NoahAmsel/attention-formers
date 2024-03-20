from itertools import islice

import matplotlib.pyplot as plt
from omegaconf import OmegaConf as oc
import pandas as pd
import torch
from tqdm import tqdm

from models import CheatingWeights
from task import dataset
from test import test_model


def loss(model, batch, scale_batch=False):
    # X has dimensions: (batch size, dim, num points)
    # Y has dimensions: (batch size, dim, num queries)
    # labels has dimensions: (batch size, dim, num queries)
    X, Y, labels = batch
    _, dim, _ = labels.shape
    labels_hat = model(X, Y)
    # MSE loss averages all the entry-wise errors
    # but we don't want to average over dimension of the vectors,
    # so mulitply by dim
    if scale_batch:
        scale = (labels * labels_hat).sum() / (labels_hat ** 2).sum()
        labels_hat *= scale
    return float(torch.nn.functional.mse_loss(labels_hat, labels) * dim)


def model_mse(model, data, num_batches, scale_batch=False):
    total_loss = 0
    for batch in tqdm(islice(iter(data), num_batches)):
        total_loss += loss(model, batch, scale_batch=scale_batch)
    return float(total_loss / num_batches)


def slope(X, Y):
    return ((X*Y).mean() - X.mean()*Y.mean()) / ((X**2).mean() - (X.mean())**2)


if __name__ == "__main__":
    config = oc.create(dict(
        dim=30,
        num_points=2,
        num_queries=1,
        task="ortho",
        scale_batch=False,
        batch_size=1024,
        num_test_batches=128,
        num_workers=0,  # this is necessary since we want to generate the dataset directly on the GPU, not in a CPU subprocess
    ))
    assert torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Hs = torch.tensor([2**i for i in range(15)])
    results = []
    for seed in range(5):
        print(f"epoch {seed}")
        for H in Hs:
            model = CheatingWeights(config.dim, H, seed=int(2e7 + seed), device=device)
            # since dataset is an iterable, must recreate it each inner
            # iteration to reset it
            data = dataset(oc.merge(config, {"seed": int(1e7+seed)}), device=device)
            squared_error = model_mse(model, data, config.num_test_batches)
            # squared_error = float(test_model(model, oc.merge(config, {"seed": int(1e7+seed)})))
            results.append((int(H), squared_error))

    df = pd.DataFrame(results, columns=["H", "Squared Error"])
    filename = f"/home/nia4240/attention-formers/lstsq_results_d={config.dim}_H={max(Hs)}_batch={config.batch_size}.csv"
    df.to_csv(filename, index=False)
    df = pd.read_csv(filename)
    curve = df.groupby("H")["Squared Error"].mean()

    print(slope(torch.log(Hs), torch.log(torch.tensor(curve.to_numpy()))))
    plt.loglog(Hs, curve)
    plt.title("Measured MSE on q,k~unif, a=C^{-1}b\ndim=2\nslope=-0.46")
    plt.xlabel("H")
    plt.ylabel("MSE")

    # import seaborn as sns
    # df["1/MSE"] = 1/df['Squared Error']
    # p = sns.lineplot(df, x='H', y='1/MSE')
    # p.set(xscale='log')
    # p.set(yscale='log')