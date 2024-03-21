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
    for N in tqdm(range(10, 100, 10)):
        for d in range(N, 200, 5):
            results.append(dict(
                num_points=N,
                dim=d,
                Pr_win=estimate_b(d, N, int(5e6))
            ))
    df = pd.DataFrame(results)
    f, ax = plt.subplots(figsize=(7, 7))
    sns.lineplot(data=df, x="dim", y="Pr_win", hue="num_points", style="num_points", ax=ax)
    ax.set(xscale="log", yscale="log")

    subplotN = 10
    df2 = df.set_index("dim")
    curve = df2.loc[df2["num_points"] == subplotN, "Pr_win"]
    s = slope(np.log(curve.index.to_numpy()), np.log(curve))
    plt.loglog(curve)
    plt.title(f"N={subplotN}, slope={s:.2f}")
    plt.xlabel("dim")
    plt.ylabel("Pr[correct]")
