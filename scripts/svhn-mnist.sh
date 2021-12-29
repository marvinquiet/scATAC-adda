#!/bin/bash

# abort entire script on error
set -e

# train base model on PBMC dataset
python tools/train.py PBMC train MLP PBMC_MLP \
       --iterations 2000 \
       --batch_size 128 \
       --display 10 \
       --lr 0.001 \
       --snapshot 500 \
       --solver adam

# run adda svhn->mnist
python tools/train_adda.py PBMC:train PBMC:test MLP adda_MLP_10X_AML \
       --iterations 2000 \
       --batch_size 50 \
       --display 10 \
       --lr 0.001 \
       --snapshot 500 \
       --weights snapshot/PBMC_MLP \
       --adversary_relu \
       --solver adam

# evaluate trained models
echo 'Source only baseline:'
python tools/eval_classification.py PBMC train MLP snapshot/PBMC_MLP

echo 'ADDA':
python tools/eval_classification.py PBMC train MLP snapshot/adda_MLP_10X_AML
