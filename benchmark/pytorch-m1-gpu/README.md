You can run these scripts as follows:


## LeNet5 on MNIST
- CPU: `python lenet-mnist.py --device "cpu"`
- NVIDIA GPU:  `python lenet-mnist.py --device "cuda"`
- Apple M1: `python lenet-mnist.py --device "mps"`

## MLP (784-256-128-64-10) on MNIST
- CPU: `python mlp-mnist.py --device "cpu"`
- NVIDIA GPU:  `python mlp-mnist.py --device "cuda"`
- Apple M1: `python mlp-mnist.py --device "mps"`

## VGG16 on CIFAR10
- CPU: `python vgg16-cifar10.py --device "cpu"`
- NVIDIA GPU:  `python vgg16-cifar10.py --device "cuda"`
- Apple M1: `python vgg16-cifar10.py --device "mps"`
