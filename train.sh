#!/bin/bash
#python -u train_lseg.py --dataset ade20k --data_path ../datasets --batch_size 4 --exp_name lseg_ade20k_l16 \
#--base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384

# clip_vitb32_384
# clip_vitl16_384

# python -u train_lseg.py --mysetup 1 --dataset ade20k --data_path ../datasets --batch_size 1 --exp_name lseg_ade20k_l16 \
# --base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitb32_384


python -u train_lseg.py --mysetup 1 --mytraintype 1  --batch_size 1 --exp_name project2_test1 \
--base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitb32_384