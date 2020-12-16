#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# CUDA_VISIBLE_DEVICES=0 ./run.sh

# General
stage=1  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES

# Arguments for generating data. For more comments, see prepare_wham_data.sh.
wham_stage=0
storage_dir=
sphere_dir=  # Directory containing sphere files
wsj0_wav_dir=
wham_wav_dir=

# Data
task=sep_clean  # Specify the task here (sep_clean, sep_noisy, enh_single, enh_both)
sample_rate=8000
mode=min
n_src=2  # If you want to train a network with 3 output streams for example.

# Training
batch_size=6
num_workers=8
#optimizer=adam
lr=0.001
epochs=100
strategy=multi_task
train_enh_dir=data/librimix/wav8k/min/train-360/

# Architecture
n_blocks=8
n_repeats=3
mask_nonlinear=relu

# Evaluation
eval_use_gpu=1


. utils/parse_options.sh

sr_string=$(($sample_rate/1000))
suffix=wav${sr_string}k/$mode
dumpdir=data/wham/$suffix  # directory to put generated json file

train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv
test_dir=$dumpdir/tt

if [[ $stage -le  0 ]]; then
  echo "Stage 0: Prepare WHAM dataset"
  . prepare_wham_data.sh \
    --stage wham_stage \
    --storage_dir $storage_dir \
    --sphere_dir $sphere_dir \
    --wsj0_wav_dir $wsj0_wav_dir \
    --wham_wav_dir $wham_wav_dir
fi

# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${task}_${sr_string}k${mode}_${uuid}
fi
expdir=exp/train_convtasnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 1 ]]; then
  echo "Stage 1: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train_general.py \
    --corpus wsj0-mix \
    --strategy $strategy \
    --train_enh_dir $train_enh_dir \
		--train_dir $train_dir \
		--valid_dir $valid_dir \
		--task $task \
		--sample_rate $sample_rate \
		--lr $lr \
		--epochs $epochs \
		--batch_size $batch_size \
		--num_workers $num_workers \
		--mask_act $mask_nonlinear \
		--n_blocks $n_blocks \
		--n_repeats $n_repeats \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log

	# Get ready to publish
  mkdir -p $expdir/publish_dir
  echo "wham/ConvTasNet" > $expdir/publish_dir/recipe_name.txt
fi

#if [[ $stage -le 2 ]]; then
	#echo "Stage 2 : Evaluation"
	#CUDA_VISIBLE_DEVICES=$id $python_path eval.py \
		#--task $task \
		#--test_dir $test_dir \
		#--use_gpu $eval_use_gpu \
		#--exp_dir ${expdir} | tee logs/eval_${tag}.log
	#cp logs/eval_${tag}.log $expdir/eval.log
#fi

