# Energy-based Variational Latent Prior (EVaLP)

This repository contains the code for implementating the experiments in paper titled "Fast and Flexible: Learning Energy-Based Variational Latent Priors for VAEs"  

## Setup  

Install the required packages  
  ``pip install -r requirements.txt``  

## Training  

### 1st Stage  

Train the VAE model  

- Change directory to vae-baseline: ``cd vae-baseline``
- Run the CelebA experiment: ``bash run_celeba.sh``
- Run the CIFAR experiment: ``bash run_cifar.sh``

### 2nd Stage  

Train the EVaLP model:  

- Change directory: ``cd 2s-vae-lebm``
- Run the CelebA experiment: ``bash run_celeba.sh``
- Run the CIFAR experiment: ``bash run_cifar.sh``

Train the NCP model:  

- Change directory: ``cd ncp-baseline``
- Run the CelebA experiment: ``bash run_celeba.sh``
- Run the CIFAR experiment: ``bash run_cifar.sh``  
  