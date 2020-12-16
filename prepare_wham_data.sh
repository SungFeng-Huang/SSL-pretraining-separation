#!/bin/bash

# Exit on error
set -e
set -o pipefail

# Main storage directory. You'll need disk space to dump the WHAM mixtures and the wsj0 wav
# files if you start from sphere files.
storage_dir=$PWD

# If you start from the sphere files, specify the path to the directory and start from stage 0
sphere_dir=$storage_dir/WSJ0  # Directory containing sphere files
# If you already have wsj0 wav files, specify the path to the directory here and start from stage 1
wsj0_wav_dir=$storage_dir/wsj0_wav
# If you already have the WHAM mixtures, specify the path to the directory here and start from stage 2
wham_wav_dir=$storage_dir/wham_wav
# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=python

# Example usage
# ./prepare_wham_data.sh --stage 0 --sphere_dir <path_to_your_wsj0> --storage_dir $PWD

# General
stage=0  # Controls from which stage to start


. utils/parse_options.sh

if [[ $stage -le  0 ]]; then
  echo "WHAM Stage 0: Converting sphere files to wav files"
  . local/wham/convert_sphere2wav.sh --sphere_dir $sphere_dir --wav_dir $wsj0_wav_dir
fi

if [[ $stage -le  1 ]]; then
	echo "WHAM Stage 1: Generating 8k and 16k WHAM dataset"
  . local/wham/prepare_data.sh --wav_dir $wsj0_wav_dir --out_dir $wham_wav_dir --python_path $python_path
fi

if [[ $stage -le  2 ]]; then
	# Make json directories with min/max modes and sampling rates
	echo "WHAM Stage 2: Generating json files including wav path and duration"
	for sr_string in 8 16; do
		for mode_option in min max; do
			tmp_dumpdir=data/wham/wav${sr_string}k/$mode_option
			echo "Generating json files in $tmp_dumpdir"
			[[ ! -d $tmp_dumpdir ]] && mkdir -p $tmp_dumpdir
			local_wham_dir=$wham_wav_dir/wav${sr_string}k/$mode_option/
      $python_path local/wham/preprocess_wham.py --in_dir $local_wham_dir --out_dir $tmp_dumpdir
    done
  done
fi
