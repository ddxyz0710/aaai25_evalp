dt=0000
vaedir=" ../vae-baseline/output/train_celeba/STD-VAE/CELEBA_WAE_PAPER_MAN_EMB_SZIE/0000"
netGpth=$vaedir/netG_bestfid.pth
netIpth=$vaedir/netI_bestfid.pth

python train_cifar.py --datetime $dt --gpu 0 --embedding_weight 0.007 --workers 4 --data_to_0_1 True --netG $netGpth --netI $netIpth 