import sys

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.cli import LightningCLI
from shared_utilities import CustomDataModule, LightningModel, PyTorchMLP
from watermark import watermark

if __name__ == "__main__":

    print(watermark(packages="torch,lightning"))

    print(f"The provided arguments are {sys.argv[1:]}")

    cli = LightningCLI(
        model_class=LightningModel,
        datamodule_class=CustomDataModule,
        run=False,
        save_config_overwrite=True,
        seed_everything_default=123,
        trainer_defaults={
            "max_epochs": 10,
            "callbacks": [ModelCheckpoint(monitor="val_acc")],
        },
    )

    pytorch_model = PyTorchMLP(num_features=100, num_classes=2)
    lightning_model = LightningModel(
        model=pytorch_model, learning_rate=cli.model.learning_rate
    )

    cli.trainer.fit(lightning_model, datamodule=cli.datamodule)
    cli.trainer.test(lightning_model, datamodule=cli.datamodule)
