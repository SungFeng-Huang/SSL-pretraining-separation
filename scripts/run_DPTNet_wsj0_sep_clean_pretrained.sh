load_path=exp/train_DPTNet_LibriMix_enh_single_from_scratch_denoise/best_model.pth
. utils/parse_options.sh

# batch_size (per GPU) = 1
# 8 GPU (V100)
# accumulate_grad_batches = 3
# total batch size = 1 * 8 * 3 = 24
bash run.sh --id 0,1,2,3,4,5,6,7 --model DPTNet --batch_size 1 --accumulate_grad_batches 3 --strategy pretrained --load_path $load_path
