#!pip install xgboost
#!pip install scikit-learn

import lightning as L
from lightning.app.storage import Drive
from my_xgboost_classifier import run_classifier


class RunCode(L.LightningWork):
    def __init__(self):

        # available GPUs and costs: https://lightning.ai/pricing/consumption-rates
        super().__init__(cloud_compute=L.CloudCompute("gpu-fast", disk_size=10))

        # storage for outputs
        self.model_storage = Drive("lit://checkpoints")

    def run(self):
        # run model code
        model_path = "my_model.joblib"
        run_classifier(save_as=model_path, use_gpu=True)
        self.model_storage.put(model_path)


component = RunCode()
app = L.LightningApp(component)