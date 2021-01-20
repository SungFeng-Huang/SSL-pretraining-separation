# batch_size (per GPU) = 3
# 8 GPU (V100)
# accumulate_grad_batches = 1
# total batch size = 3 * 8 * 1 = 24
bash run.sh --id 0,1,2,3,4,5,6,7 --model DPRNNTasNet --batch_size 3 --segment 2
