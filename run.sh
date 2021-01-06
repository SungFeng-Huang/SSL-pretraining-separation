#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# bash ./run.sh --id 0 
# bash ./run.sh --id 0 --strategy pretrained --load_path exp/xxx/model.pth
# bash ./run.sh --id 0 --strategy multi_task --enh_set train-360

# General
stage=1  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES
corpus=wsj0-mix # wsj0-mix or LibriMix
model=ConvTasNet  # The model class

# Arguments for generating data. For more comments, see prepare_wham_data.sh.
storage_dir=
if [[ $corpus == "wsj0-mix" ]]; then
  wham_stage=0
  sphere_dir=  # Directory containing sphere files
  wsj0_wav_dir=
  wham_wav_dir=
fi

# Data
task=sep_clean  # one of 'enh_single', 'enh_both', 'sep_clean', 'sep_noisy'
sample_rate=8000
mode=min
n_src=2
segment=4

# Training config
epochs=100
batch_size=6 # batch size per step
accumulate_grad_batches=1  # accumulate steps
num_workers=8
half_lr=yes
early_stop=yes
strategy=from_scratch
load_path=
enh_set=train-360
resume=no
comet=yes
comet_exp_key=
resume_ckpt=

# Optim config
optimizer=adam
lr=0.001
weight_decay=0.

# Network config
if [[ $model == "ConvTasNet" ]]; then
  n_blocks=8
  n_repeats=3
  mask_nonlinear=relu
else
  # Add whatever config you want to modify here, and also modify $train_cmd
  # below.
  true
fi

# Data config
train_set=train-100
valid_set=dev
test_set=test

# Evaluation
eval_use_gpu=1


. utils/parse_options.sh


sr_string=$(($sample_rate/1000))
suffix=wav${sr_string}k/$mode

if [[ $corpus == "LibriMix" ]]; then
  dumpdir=data/librimix/$suffix  # directory to put generated csv file
  train_dir=$dumpdir/$train_set
  valid_dir=$dumpdir/$valid_set
  test_dir=$dumpdir/$test_set
elif [[ $corpus == "wsj0-mix" ]]; then
  dumpdir=data/wham/$suffix  # directory to put generated json file
  train_dir=$dumpdir/tr
  valid_dir=$dumpdir/cv
  test_dir=$dumpdir/tt
fi

if [[ $stage -le  0 ]]; then
	echo "Stage 0: Generating $corpus dataset"
  if [[ $corpus == "LibriMix" ]]; then
    . prepare_librimix_data.sh --storage_dir $storage_dir --n_src $n_src
  elif [[ $corpus == "wsj0-mix" ]]; then
    . prepare_wham_data.sh --stage wham_stage --storage_dir $storage_dir \
      --sphere_dir $sphere_dir --wsj0_wav_dir $wsj0_wav_dir --wham_wav_dir $wham_wav_dir
  fi
fi

# Generate a random ID for the run if no tag is specified
# May need a better recognizable automatic tag in the future
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
	tag=${uuid}
fi

expdir=exp/train_${model}_${corpus}_${task}_${strategy}_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"


# Remove those you want to use from yaml instead of here
train_cmd="--corpus $corpus --model $model \
  --train_dir $train_dir --valid_dir $valid_dir \
  --task $task --sample_rate $sample_rate --n_src $n_src --segment $segment \
  --epochs $epochs --batch_size $batch_size --accumulate_grad_batches $accumulate_grad_batches \
  --num_workers $num_workers --half_lr $half_lr --early_stop $early_stop \
  --optimizer $optimizer --lr $lr --weight_decay $weight_decay"

# Training config
if [[ $strategy == "multi_task" && -n $enh_set ]]; then
  dumpdir=data/librimix/$suffix  # directory to put generated csv file
  train_enh_dir=$dumpdir/$enh_set
  train_cmd="$train_cmd --strategy $strategy --train_enh_dir $train_enh_dir"
elif [[ $strategy == "pretrained" && -n $load_path ]]; then
  train_cmd="$train_cmd --strategy $strategy --load_path $load_path"
fi

if [[ $comet == "yes" ]]; then
  train_cmd="$train_cmd --comet"
fi
if [[ $resume == "yes" ]]; then
  train_cmd="$train_cmd --resume"
  if [[ -n $resume_ckpt ]]; then
    train_cmd="$train_cmd --resume_ckpt $resume_ckpt"
  fi
  if [[ -n $comet_exp_key ]]; then
    train_cmd="$train_cmd --comet_exp_key $comet_exp_key"
  fi
fi

# Network config
if [[ $model == "ConvTasNet" ]]; then
  train_cmd="$train_cmd --n_blocks $n_blocks --n_repeats $n_repeats --mask_act $mask_nonlinear"
fi


if [[ $stage -le 1 ]]; then
  echo "Stage 1: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train_general.py \
    $train_cmd \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log

	# Get ready to publish
  # NOTE: Not recommend to publish from this repo, the recipe_name would be
  # confusing. If you wish to upload your pretrained models, please directly run
  # the code from asteroid official repo and follow the upload guideline.
  mkdir -p $expdir/publish_dir
  echo "SungFeng-Huang/SSL-pretraining-separation" > $expdir/publish_dir/recipe_name.txt
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
