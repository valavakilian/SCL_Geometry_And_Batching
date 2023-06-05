# Convergence of SCL to Orthogonal Frame
In order to reproduce logs please run the following command:

**CIFAR10 + ResNet18**
```bash
python main.py --gpu --loss_type SCL --model ResNet18 --dataset CIFAR10
```

**CIFAR10 + DenseNet**
```bash
python main.py --gpu --loss_type SCL --model DenseNet --dataset CIFAR10
```

Similar change ``` --dataset ``` to ``` MNSIT, FMNIST, CIFAR100 ```. As well, other hyperparametrs can be controlled with arguments included in the main file.

One experiments are completem models will be saved in ./logs_model and logs will be saved in ./logs on default. 

Following this, a range of results can be generated with the code in ./graph to generate figures from paper.

Further instructions will be provided upong further updates. Thank you for your understanding.
