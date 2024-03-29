from itertools import islice
from lightning import seed_everything
from task import *

# in create_rng, comment out all but first and last line
# in __iter__, add second line: print(f"Creating iter with initial seed: {rng.initial_seed()}\t seed {rng.seed()}")


seed_everything(0, workers=True)
data_loader = dataset("unif", 2, 1, 1, 2, num_workers=2, seed=None)

first4 = list(islice(iter(data_loader), 4))

def first_coord_of_x1(batch):
    # batch[0] means X
    # X is batch_size x dim x num points 
    return batch[0][:, 0, 0]

for batch in first4:
    print(first_coord_of_x1(batch))


# print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))