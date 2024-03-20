from pathlib import Path

from omegaconf import OmegaConf as oc
import pandas as pd


def experiment_analysis(csv_logs_path):
    run_folders = Path("csv_logs/test_sweep").glob('**/version_*')
    run_data = []
    for run_folder in run_folders:
        metrics = pd.read_csv(Path(run_folder, "metrics.csv"))
        config = oc.load(Path(run_folder, "config.yaml"))
        run_data.append(dict(
            loss=metrics.train_loss.min(),
            lr=metrics.loc[0, "lr-SGD"],
            rank=config.model.rank,
            nheads=config.model.nheads,
            dim=config.data.dim,
            num_points=config.data.num_points,
        ))
    return pd.DataFrame(run_data)
