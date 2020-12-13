#!/bin/bash

# Exit on error
set -e
set -o pipefail

# If you haven't generated LibriMix start from stage 0
# Main storage directory. You'll need disk space to store LibriSpeech, WHAM noises
# and LibriMix. This is about 500 Gb
storage_dir=

# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=1  # Controls from which stage to start
tag="DPTNet_wsj0-2mix_sep_clean"  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0,1,2,3,4,5,6,7
#id=$CUDA_VISIBLE_DEVICES
out_dir=wsj0-2mix # Controls the directory name associated to the evaluation results inside the experiment directory

# Network config
# Training config
epochs=200
batch_size=24
num_workers=8
half_lr=yes
early_stop=yes
# Optim config
optimizer=adam
lr=0.001
weight_decay=0.00001
# Data config
train_dir=wsj0-mix/2speakers/wav8k/min/tr
valid_dir=wsj0-mix/2speakers/wav8k/min/cv
test_dir=wsj0-mix/2speakers/wav8k/min/tt
sample_rate=8000
n_src=2
segment=4
task=sep_clean  # one of 'enh_single', 'enh_both', 'sep_clean', 'sep_noisy'

. utils/parse_options.sh


if [[ $stage -le  0 ]]; then
    echo "Stage 0: Generating Librimix dataset"
    . local/prepare_data.sh --storage_dir $storage_dir --n_src $n_src
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
    tag=${uuid}
fi

expdir=exp/train_dptnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

#loadpth=None
loadpth=pretrained_ckpt/ConvTasNet_Libri1Mix_enhsingle.pth


if [[ $stage -le 1 ]]; then
    echo "Stage 1: Training"
    mkdir -p logs
    #--load_path $loadpth \
        #--multi_task \
        #--train_enh_dir data/wav8k/min/train-360/ \
    CUDA_VISIBLE_DEVICES=$id $python_path train_general.py --exp_dir $expdir \
        --corpus wsj0-mix \
        --model DPTNet \
        --real_batch_size 8 \
        --epochs $epochs \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --half_lr $half_lr \
        --early_stop $early_stop \
        --optimizer $optimizer \
        --lr $lr \
        --weight_decay $weight_decay \
        --train_dir $train_dir \
        --valid_dir $valid_dir \
        --sample_rate $sample_rate \
        --n_src $n_src \
        --task $task \
        --segment $segment | tee logs/train_${tag}.log
    cp logs/train_${tag}.log $expdir/train.log

    # Get ready to publish
    mkdir -p $expdir/publish_dir
    echo "librimix/DPTNet" > $expdir/publish_dir/recipe_name.txt
fi

#if [[ $stage -le 2 ]]; then
#    echo "Stage 2 : Evaluation"
#    $python_path eval.py --exp_dir $expdir --test_dir $test_dir \
#        --out_dir $out_dir \
#        --task $task | tee logs/eval_${tag}.log
#    cp logs/eval_${tag}.log $expdir/eval.log
#fi
