# batch_size (per GPU) = 1
# 8 GPU (V100)
# accumulate_grad_batches = 3
# total batch size = 1 * 8 * 3 = 24
bash run.sh --id 0,1,2,3,4,5,6,7 --corpus LibriMix --model DPTNet --batch_size 1 --accumulate_grad_batches 3 --tag denoise --n_src 1 --task enh_single
