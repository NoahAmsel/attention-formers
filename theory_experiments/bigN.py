import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from kernel_computations import HeadVsTarget, samples_from_sphere, slope


def estimate_b(dim, N, num_trials):
    assert N <= dim
    num_ys = int(np.ceil(np.sqrt(num_trials)))
    num_qs = num_ys
    ys = samples_from_sphere(dim, num_ys)
    qs = samples_from_sphere(dim, num_qs)
    truth = np.argmax(ys[:N, :], axis=0)  # num_ys
    heads = np.argmax(qs[:N, :].reshape(N, num_qs, 1) * (qs.T @ ys).reshape(1, num_qs, num_ys), axis=0) # num qs, num ys
    return (heads == truth).mean()


if __name__ == "__main__":
    # check correctness for N=2
    dims = list(range(3, 20))
    old = [HeadVsTarget(d, 100)(0) for d in dims]
    new = [estimate_b(d, 2, int(1e6)) for d in dims]
    plt.plot(dims, old, dims, new)


if __name__ == "__main__":
    # p(win) vs N
    dim = 15
    Ns = np.array(list(range(2, 12)))
    bs = np.array([estimate_b(dim, N, int(5e6)) for N in tqdm(Ns)])
    s = slope(Ns, 1/bs)
    plt.plot(Ns, 1/bs)
    plt.title(f"Probability of uniform random q=k head being correct\nslope: {s:.2}")
    plt.xlabel("Number of points x_1, ... x_N")
    plt.ylabel("1/Pr[correct]")


if __name__ == "__main__":
    # p(win) vs d
    results = []
    for N in tqdm(np.round(np.geomspace(10, 100, num=10)).astype(int)):
    # for N in [10]:
        for d in np.round(np.geomspace(N, 100*N, num=10)).astype(int):
            results.append(dict(
                num_points=N,
                dim=d,
                Pr_win=estimate_b(d, N, int(5e6))
            ))
    df = pd.DataFrame(results)
    df["edge"] = df["Pr_win"] - 1/df["num_points"]
    f, ax = plt.subplots(figsize=(7, 7))
    sns.lineplot(data=df, x="dim", y="edge", hue="num_points", style="num_points", ax=ax)
    ax.set(xscale="log", yscale="log")

if __name__ == "__main__":
    N = 10
    ds = np.round(np.geomspace(N, 100 * N, num=20)).astype(int)
    bs = np.array([estimate_b(d, N, int(5e6)) for d in ds])
    s = slope(np.log(ds), np.log(bs))
    joan = (1/N + 1/np.sqrt(ds))
    plt.loglog(ds, bs, marker='.', label="empirical")
    plt.loglog(ds, joan, label="1/N + 1/sqrt(d)")
    plt.hlines([1/N], min(ds), max(ds), label="1/N")
    plt.title(f"N={N}, slope={s:.2f}")
    plt.xlabel("dim")
    plt.ylabel("Pr[correct]")

    edge = bs - (1/N)
    edge_slope = slope(np.log(ds), np.log(edge))
    plt.loglog(ds, edge, marker=".")
    plt.title(f"N={N}, slope={edge_slope:.2f}")
    plt.xlabel("dim")
    plt.ylabel("edge = Pr[correct] - 1/N")

if __name__ == "__main__":
    dim = 100
    Ns = np.round(np.geomspace(dim//10, dim, num=20)).astype(int)
    bs = np.array([estimate_b(dim, N, int(5e6)) for N in tqdm(Ns)])
    edge = bs - (1/Ns)
    edge_slope = slope(np.log(Ns), np.log(edge))
    plt.loglog(Ns, edge, marker=".")
    plt.title(f"dim={dim}, slope={edge_slope:.2f}")
    plt.xlabel("N")
    plt.ylabel("edge = Pr[correct] - 1/N")
