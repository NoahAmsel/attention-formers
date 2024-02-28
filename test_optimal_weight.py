from itertools import islice

import matplotlib.pyplot as plt
from omegaconf import OmegaConf as oc
import pandas as pd
import torch
from tqdm import tqdm

from models import OptimallyWeightedRandom
from task import dataset
from train import test_model


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
    return torch.nn.functional.mse_loss(labels_hat, labels) * dim


def model_mse(model, data, num_batches, scale_batch=False):
    total_loss = 0
    for batch in tqdm(islice(iter(data), num_batches)):
        total_loss += loss(model, batch, scale_batch=scale_batch)
    return float(total_loss / num_batches)


def slope(X, Y):
    return ((X*Y).mean() - X.mean()*Y.mean()) / ((X**2).mean() - (X.mean())**2)


if __name__ == "__main__":
    config = oc.create(dict(
        dim=2,
        num_points=2,
        num_queries=4,
        task="ortho",
        scale_batch=False,
        batch_size=64,
        num_test_batches=128,
        num_workers=1,
    ))
    Hs = torch.tensor([2**i for i in range(15)])
    results = []
    for seed in range(10):
        print(f"epoch {seed}")
        for H in Hs:
            model = OptimallyWeightedRandom(H, seed=int(2e7 + seed))
            # since dataset is an iterable, must recreate it each inner
            # iteration to reset it
            data = dataset(oc.merge(config, {"seed": int(1e7+seed)}))
            squared_error = model_mse(model, data, config.num_test_batches)
            # squared_error = float(test_model(model, oc.merge(config, {"seed": int(1e7+seed)})))
            results.append((int(H), squared_error))

    df = pd.DataFrame(results, columns=["H", "Squared Error"])
    # df.to_csv("results.csv", index=False)
    # df = pd.read_csv("results.csv")
    curve = df.groupby("H")["Squared Error"].mean()

    print(slope(torch.log(Hs), torch.log(torch.tensor(curve.to_numpy()))))
    plt.loglog(Hs, curve)
    plt.title("Measured MSE on q,k~unif, a=C^{-1}b\ndim=2\nslope=-0.46")
    plt.xlabel("H")
    plt.ylabel("MSE")
