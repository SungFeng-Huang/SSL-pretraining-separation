#!/bin/bash

# Main storage directory. You'll need disk space to store LibriSpeech, WHAM noises
# and LibriMix. This is about 472GB for Libri2Mix and 369GB for Libri3Mix
storage_dir=
n_src=
python_path=python

. ./utils/parse_options.sh

current_dir=$(pwd)
# Clone LibriMix repo
git clone https://github.com/JorisCos/LibriMix

# Run generation script
# Modify generate_librimix.sh if you only want to generate a subset of LibriMix
cd LibriMix
. generate_librimix.sh $storage_dir

cd $current_dir
$python_path local/librimix/create_local_metadata.py --librimix_dir $storage_dir/Libri$n_src"Mix"
