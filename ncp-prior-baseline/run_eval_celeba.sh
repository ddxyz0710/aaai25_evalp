dt=`date +%Y-%m-%d-%H-%M-%S`
vaedir=" ../vae-baseline/output/train_celeba/STD-VAE/CELEBA_WAE_PAPER_MAN_EMB_SZIE/2024-07-22-19-37-03/"
netGpth=$vaedir/netG_bestfid.pth
netIpth=$vaedir/netI_bestfid.pth

#-======  EVAL ============
# If eval was completed during training, no need to run the following code again. 

outdir='output/train_celeba/STD-VAE/CELEBA_WAE_PAPER_MAN_EMB_SZIE/2024-07-28-04-30-54'
netEpth=$outdir/netE_bestbce.pth
python -m pdb train_celeba.py --mode 'eval' --datetime $dt --gpu 1 --workers 4 \
    --data_to_0_1 True --netG $netGpth --netI $netIpth --netE $netEpth \
    --generate True