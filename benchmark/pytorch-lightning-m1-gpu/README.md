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



Recommended: upgrade PyTorch and PyTorch Lightning to the latest versions, e.g., 

```
pip install torch --upgrade
pip install pytorch_lighting --upgrade
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



4 GPUs

```
python main.py --output_path results \
--mixed_precision false \
--num_epochs 3 \
--batch_size 256 \
--num_epochs 3 \
--num_devices 4 \
--log_accuracy false \
--accelerator gpu \
--strategy ddp_spawn
```

Training time: 2.20 min
Inference time (test set): 0.32 min

---



1 GPU

```
python main.py --output_path results \
--mixed_precision false \
--num_epochs 3 \
--batch_size 128 \
--num_epochs 3 \
--num_devices 1 \
--log_accuracy false \
--accelerator gpu \
```

Training time: 6.47 min
Inference time (test set): 0.11 min

---



Multi-CPU with `ddp_spawn`

```
python main.py --output_path results \
--mixed_precision false \
--num_epochs 3 \
--batch_size 256 \
--num_epochs 3 \
--num_devices auto \
--log_accuracy false \
--accelerator cpu \
--strategy ddp_spawn
```

Training time: 
Inference time (test set):

---



1 CPU

```
python main.py --output_path results \
--mixed_precision false \
--num_epochs 3 \
--batch_size 256 \
--num_epochs 3 \
--log_accuracy false \
--num_devices 1 \
--accelerator cpu \
```

Training time: 
Inference time (test set):

---



## RTX 2080Ti

python main.py --output_path results \
--mixed_precision false \
--num_epochs 3 \
--batch_size 128 \
--num_epochs 3 \
--device_numbers 1,2,3,5 \
--log_accuracy false \
--accelerator gpu \
--strategy ddp_spawn

1.56 min

0.38

python main.py --output_path results \
--mixed_precision true \
--num_epochs 3 \
--batch_size 128 \
--num_epochs 3 \
--device_numbers 1,2,3,5 \
--log_accuracy false \
--accelerator gpu \
--strategy ddp_spawn

1.42 min

0.44

python main.py --output_path results \
--mixed_precision true \
--num_epochs 3 \
--batch_size 128 \
--num_epochs 3 \
--num_devices 1 \
--log_accuracy false \
--accelerator gpu \
--strategy ddp_spawn



## M1 Pro

TBD
