# Learning the Space of Deep Models

Official code for the paper "*Learning the Space of Deep Models*", published at ICPR 2022.  
Authors: Gianluca Berardi*, Luca De Luigi*, Samuele Salti, Luigi Di Stefano.  
\* joint first authorship

## Setting up the environment
We conducted our experiments using Ubuntu 18.04.5 and python 3.6.9.  
The python environment used in our experiments can be created with the following commands:

```
sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Images and SDF datasets
NetSpace is trained on networks that are trained either to classify images or to represent implicitly
the SDF of 3D shapes.  
As far as images are concerned, we used CIFAR-10 and TinyImageNet. For CIFAR-10 we used the version
provided by `torchvision`, so it will be downloaded automatically the first time.
TinyImageNet must be instead downloaded manually and placed in the directory `data\tiny-imagenet-200`.
Please note that we created a validation split with images from the training set: the split files that
we used are in the directory `data\tiny-imagenet-200` and the images should be arranged accordingly.  
For what concerns SDF, we used the code provided by [DeepSDF](https://github.com/facebookresearch/DeepSDF)
to compute the SDF for the class `chair`.

## Training NetSpace
We provide several python scripts to run the experiments reported in our paper.
Each python file contains several parameters that need to be changed, to indicate output dirs,
dataset paths, etc. Be sure to set all the parameters properly before running the python files.

### Creating the datasets of networks
The datasets of networks can be created by running the commands:

```
python3 create_nets_dataset_single.py (Single-Architecture Image classification)
python3 create_mlp_dataset.py (Single-Architecture SDF regression)
python3 create_nets_dataset_multi.py (Multi-Architecture Image classification)
```

### Creating input list files
Once that all the required instances have been trained, for the Single-Architecture Image
classification and the Multi-Architecture experiments it's necessary to create the input lists
for each setting/architecture. See files in the directory `models/input_lists` as example.

### Singe-Architecture (Image classification) training 
The Single-Architecture (Image classification) training can be executed with the command:

```
python3 train_single_arch.py
```

### Single-Architecture (SDF regression) training
The Single-Architecture (SDF regression) training can be executed with the command:

```
python3 train_mlps.py
```

### Multi-Architecture training
The Multi-Architecture training can be executed with the command:

```
python3 train_multi_arch.py
```

The file `train_multi_arch.py` contains the parameters that can be edited to select
whether the training set should contain all the architectures (i.e. paper Sec. 4
"Multi-Architecture") or only LeNetLike and ResNet32 (i.e. paper Sec. 4 "Sampling of Unseen
Architectures").

### Latent Space Optimization
The Latent Space Optimization training can be executed with the commands:

```
python3 train_latspace_single.py
python3 train_latspace_multi.py
```

The file `train_latspace_single.py` deals with latent space optimization starting from a 
Single-Architecture Image classification training, while `train_latspace_multi.py` concerns
latent space optimization starting from a Multi-Architecture training.

## Evaluation
To produce the evaluation results reported in the paper, it is possibile to run the following commands:

```
python3 eval_single.py
python3 interpolate_single_arch.py
python3 interpolate_mlps.py
python3 eval_multi.py
python3 interpolate_multi_arch.py
python3 eval_latspace_single.py
python3 eval_latspace_multi.py
python3 visualize_prep.py
```