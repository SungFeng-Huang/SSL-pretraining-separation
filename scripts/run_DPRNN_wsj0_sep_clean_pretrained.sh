load_path=exp/train_DPRNNTasNet_LibriMix_enh_single_from_scratch_denoise/best_model.pth
. utils/parse_options.sh

# 8 GPU (V100), total batch size=24
bash run.sh --id 0,1,2,3,4,5,6,7 --model DPRNNTasNet --batch_size 24 --segment 2 --strategy pretrained --load_path $load_path
