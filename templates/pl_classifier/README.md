# Classifier Project Template



This is a classifier template code for re-use. In this specific instance, it's MobileNet v3 (large) on CIFAR-10 (rescaled to ImageNet size, 224x224).



I recommend setting up this project as follows:



## 1 - Set up a fresh environment

```bash
conda create -n clf-template python=3.8     
conda activate clf-template
```



## 2 - Install project requirements


```bash
pip install -r requirements.txt
```



## 3 - Install utility code as a Python package

This is optional and only required if you want to run the code outside this reposistory.

Assuming you are inside this folder, run

```bash
pip install -e .
```



## 4 - Inspect the Dataset



Run the notebook [./notebooks/4_inspecting-the-dataset.ipynb](./notebooks/4_inspecting-the-dataset.ipynb).



## 5 - Run the Main Training Script


Run the [main.py](main.py) code as follows, e.g., on a server:

```bash
python main.py --output_path my-results \
--mixed_precision true \
--num_epochs 10 \
--batch_size 128 \
--learning_rate 0.0005 \
--num_epochs 10 \
--accelerator gpu \
--num_devices 4 \
--strategy ddp_spawn
--log_accuracy true \
```

- Run this script with different hyperparameter settings.
- You can change `--num_devices` to `"auto"` to utilize all GPUs on the given machine. 






## 6 - Inspect the results

Run the notebook [./notebooks/6_inspecting-the-dataset.ipynb](./notebooks/6_evaluating-the-results.ipynb).



## 7 - Iterate

- Repeat steps 4-7 with modified datasets, models, and so forth.



## 8 - Use the Final Model

- See the [Inference in Production](https://pytorch-lightning.readthedocs.io/en/stable/common/production_inference.html) docs for your use case.



