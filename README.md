# SSL-pretraining-separation
This is the official repository of [SELF-SUPERVISED PRE-TRAINING REDUCES LABEL PERMUTATION INSTABILITY OF SPEECH SEPARATION
](https://arxiv.org/pdf/2010.15366.pdf), which is not fully organized yet.

------------------------------------
Corpus Preprocessing
------------------------------------
### WHAM! / WSJ0-mix
- Prepare your WSJ0 corpus and place under `./`
- Run:
```bash
bash prepare_wham_data.sh
```

### Libri2Mix
- Run:
``` bash
bash prepare_librimix_data.sh --n_src 2
```

------------------------------------
Train
------------------------------------
### Example 1: training from scratch
- Conv-TasNet (default)
- WSJ0-2mix (default)
- multi-GPU, CUDA_VISIBLE_DEVICES=0,1
```bash
# CUDA_VISIBLE_DEVICES could be passed to --id
bash ./run.sh --id 0,1 \
              --model ConvTasNet \
              --corpus wsj0-mix \
              --strategy from_scratch
```

### Example 2: pre-training
- Conv-TasNet (default)
- task: enh_single
  - means speech enhancement/denoise with only single clean speaker
- Libri1Mix train-360
  - Libri1Mix is actually Libri2Mix, but only speaker1 is used to mix with noise
- n_src: 1 (only output single speaker)
- multi-GPU, CUDA_VISIBLE_DEVICES=0,1,2,3
```bash
# CUDA_VISIBLE_DEVICES could also be specified in front
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./run.sh --model ConvTasNet \
                                           --corpus LibriMix \
                                           --task enh_single \
                                           --n_src 1 \
                                           --train_set train-360 \
                                           --strategy from_scratch
```

### Example 3: fine-tuning
- DPRNN
- Libri2Mix
- pretrained checkpoint: exp/xxx/model.pth
- single-GPU, CUDA_VISIBLE_DEVICES=0
```bash
bash ./run.sh --id 0 \
              --model DPRNNTasNet \
              --corpus LibriMix \
              --strategy pretrained \
              --load_path exp/xxx/model.pth
```

### Example 4: multi-task training
- DPTNet
- WSJ0-2mix (default)
- multi-task training
- speech denoise/enhancement data path: `data/librimix/wav8k/min/train-360/`
- multi-GPU, CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
- total batch size: 8
- batch size on each gpu: 1 (need to accumulate gradient for 2 times per update)
```bash
bash ./run.sh --id 0,1,2,3,4,5,6,7 \
              --model DPTNet \
              --strategy multi_task \
              --enh_set train-360 \
              --batch_size 8 \
              --real_batch_size 4
```

------------------------------------
Organizing Progress
------------------------------------
### Corpus Preprocessing
* [x] WSJ0-2mix
* [x] Libri2Mix

### Models
* [x] ConvTasNet
* [x] DPRNNTasNet
* [x] DPTNet

### Training
* [x] WSJ0-2mix
* [x] Libri2Mix

### Evaluation
* [ ] WSJ0-2mix
* [ ] Libri2Mix

------------------------------------
Reference
------------------------------------
The codes were adapted from
- [asteroid/egs/librimix/ConvTasNet/](https://github.com/asteroid-team/asteroid/tree/master/egs/librimix/ConvTasNet)
- [asteroid/egs/wham/ConvTasNet/](https://github.com/asteroid-team/asteroid/tree/master/egs/wham/ConvTasNet).
