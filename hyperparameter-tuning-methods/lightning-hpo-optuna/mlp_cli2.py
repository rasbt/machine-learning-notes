import sys

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger
from shared_utilities import CustomDataModule, LightningModel2
from watermark import watermark

if __name__ == "__main__":

    print(watermark(packages="torch,lightning"))

    print(f"The provided arguments are {sys.argv[1:]}")

    cli = LightningCLI(
        model_class=LightningModel2,
        datamodule_class=CustomDataModule,
        run=False,
        save_config_overwrite=True,
        seed_everything_default=123,
        trainer_defaults={
            "logger": CSVLogger(save_dir="sweep-logs/", name="my-sweep"),
            "callbacks": [ModelCheckpoint(monitor="val_acc")],
        },
    )

    print(cli.model.hidden_units)

    lightning_model = LightningModel2(
        model=None,
        hidden_units=cli.model.hidden_units,
        learning_rate=cli.model.learning_rate,
    )

    cli.trainer.fit(lightning_model, datamodule=cli.datamodule)
    #cli.trainer.test(lightning_model, datamodule=cli.datamodule)
