# 8 GPU (V100), total batch size=24
bash run.sh --id 0,1,2,3,4,5,6,7 --model DPTNet --batch_size 8 --accumulate_grad_batches 3
