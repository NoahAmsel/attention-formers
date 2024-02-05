import numpy as np
from tqdm import tqdm

dim = 20
theta = np.pi/4
q = np.zeros(dim)
q[0] = 1
k = np.zeros(dim)
k[:2] = [np.cos(theta), np.sin(theta)]

count1 = 0
count2 = 0
ntrials = 100_000
for _ in tqdm(range(ntrials)):
    r = np.random.randn(dim, 3)
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    common = np.inner(x, k) * np.inner(q, y)
    if common * np.inner(x, z) * np.inner(z, y) > 0:
        count1 += 1
    if common * np.inner(x, y) > 0:
        count2 += 1
# print(count1 / ntrials)
# print(count2 / ntrials)
print(count1 / count2)
