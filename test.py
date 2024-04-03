import lightning as L
from lightning.pytorch.cli import ArgsType, LightningCLI
import torch

from task import NearestPointDataModule, NearestPointDatasetOrthogonal
from train import LitSequenceRegression


def test_model(model_module, data_module, num_test_batches, scale_batch=False):
    lit_model = LitSequenceRegression(model_module, scale_batch=scale_batch)
    tester = L.Trainer(limit_test_batches=num_test_batches, logger=False)
    return tester.test(model=lit_model, dataloaders=data_module)


class ZeroModel(torch.nn.Module):
    def forward(self, X, Y):
        return torch.zeros_like(Y)


class MyTestCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.dim", "model.model.init_args.dim")
        # TODO: something to make it easier to configure batch size separately from limit_test_batches


def test_main(args: ArgsType = None):
    cli = MyTestCLI(
        LitSequenceRegression,
        NearestPointDataModule,
        save_config_callback=None,
        trainer_defaults=dict(logger=False),
        seed_everything_default=613,
        parser_kwargs={"parser_mode": "omegaconf"},
        args=args,
        run=False,
    )
    return cli.trainer.test(cli.model, cli.datamodule)


if __name__ == "__main__":
    test_main()

    # Example command line call:
    # python new_test.py --model.model=ZeroModel --data.dataset_class=task.NearestPointDatasetOrthogonal --data.dim=16 --data.num_points=2 --data.num_queries=1 --data.batch_size=512 --data.num_workers=3 --trainer.limit_test_batches=32

    # Alternative ways to call main:

    # test_main({
    #     "model.model": "ZeroModel",
    #     "data.dataset_class": NearestPointDatasetOrthogonal,
    #     "data.dim": 16,
    #     "data.num_points": 2,
    #     "data.num_queries": 1,
    #     "data.batch_size": 512,
    #     "data.num_workers": 3,
    #     "trainer.limit_test_batches": 32
    # })

    # test_main(dict(
    #     # model=dict(model=dict(class_path="OptimallyWeightedRandom", init_args=dict(nheads=int(2**10), num_gegenbauer_terms=30, scipy_solver=True))),
    #     # model=dict(model=dict(class_path="RandomQKEqual", init_args=dict(rank=1, nheads=128))),
    #     # model=dict(model=dict(class_path="EqualSpacing2D", init_args=dict(nheads=128))),
    #     model=dict(model="PerfectFullRank"),
    #     data=dict(
    #         dataset_class=NearestPointDatasetOrthogonal,
    #         dim=16,
    #         num_points=2,
    #         num_queries=1,
    #         batch_size=512,
    #         num_workers=3
    #     ),
    #     trainer=dict(
    #         limit_test_batches=32
    #     ),
    # ))
