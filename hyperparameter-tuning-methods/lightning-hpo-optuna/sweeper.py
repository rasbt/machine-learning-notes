import os.path as ops

import optuna
from lightning import LightningApp

from lightning_hpo import Sweep
from lightning_hpo.algorithm.optuna import OptunaAlgorithm
from lightning_hpo.distributions.distributions import Categorical, IntUniform, LogUniform

app = LightningApp(
    Sweep(
        script_path=ops.join(ops.dirname(__file__), "./mlp_cli2.py"),
        n_trials=3,
        distributions={
            "model.learning_rate": LogUniform(0.001, 0.1),
            "model.hidden_units": Categorical(["[50, 100]", "[100, 200]"]),
            "data.batch_size": Categorical([32, 64]),
            "trainer.max_epochs": IntUniform(1, 3),
        },
        algorithm=OptunaAlgorithm(optuna.create_study(direction="maximize")),
        framework="pytorch_lightning",
    )
)