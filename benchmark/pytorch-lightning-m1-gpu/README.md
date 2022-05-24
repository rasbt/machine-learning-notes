This is some simple benchmark code for evaluating MobileNet v3 (large) on CIFAR-10 (rescaled to ImageNet size, 224x224).



You can set up the experiments as follows:



## 1 Set up a fresh environment

```
conda create -n clf-template python=3.8     
conda activate clf-template
```



## 2 Install requirements


```
pip install -r requirements.txt
```



## 3 Install as Python Package

This is optional and only required if you want to run the code outside this reposistory

Assuming you are inside this folder, run

```
pip install -e .
```



## 4 Install Nightly Releases with M1 GPU support


TBD



# Benchmark results



You can run the following codes to replicate the benchmarks.



## GTX 1080Ti

On a workstation with 4 x GTX 1080Ti cards and Intel Xeon E5-2650 (12 core)

```
python main.py --output_path results \\
--mixed_precision false \\
--num_epochs 10 \\
--batch_size 256 \\
--num_epochs 10 \\
--num_devices 4 \\
--accelerator "gpu" \\
--strategy "ddp_spawn"
```

Training time: 
Inference time (test set):

---

On a workstation with 4 x GTX 1080Ti cards and Intel Xeon E5-2650 (12 core)

```
python main.py --output_path results \\
--mixed_precision false \\
--num_epochs 10 \\
--batch_size 256 \\
--num_epochs 10 \\
--num_devices 1 \\
--accelerator "gpu" \\
--strategy "ddp_spawn"
```

Training time: 
Inference time (test set):

---

On a workstation with 4 x GTX 1080Ti cards and Intel Xeon E5-2650 (12 core)

```
python main.py --output_path results \\
--mixed_precision false \\
--num_epochs 10 \\
--batch_size 256 \\
--num_epochs 10 \\
--num_devices 1 \\
--accelerator "gpu" \\
--strategy "ddp_spawn"
```

Training time: 
Inference time (test set):

---

On a workstation with 4 x GTX 1080Ti cards and Intel Xeon E5-2650 (12 core)

```
python main.py --output_path results \\
--mixed_precision false \\
--num_epochs 10 \\
--batch_size 256 \\
--num_epochs 10 \\
--num_devices "auto" \\
--accelerator "cpu" \\
--strategy "ddp_spawn"
```

Training time: 
Inference time (test set):

---

On a workstation with 4 x GTX 1080Ti cards and Intel Xeon E5-2650 (12 core)

```
python main.py --output_path results \\
--mixed_precision false \\
--num_epochs 10 \\
--batch_size 256 \\
--num_epochs 10 \\
--num_devices 1 \\
--accelerator "cpu" \\
```

Training time: 
Inference time (test set):

---

## RTX 2080Ti

TBD

## M1 Pro

TBD
