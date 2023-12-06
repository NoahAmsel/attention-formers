from omegaconf import OmegaConf as oc

from models import HardMultiheadAttention
from train import test

if __name__ == "__main__":
    config = oc.create(dict(
        dim=3,
        num_points=4,
        num_queries=5,
        double_points=False,
        batch_size=2,
        num_workers=4,
        num_batches=100
    ))
    model = HardMultiheadAttention.farthest_init(config.dim)
    test(model, config)
