# SSL-pretraining-separation
This is the official repository of [SELF-SUPERVISED PRE-TRAINING REDUCES LABEL PERMUTATION INSTABILITY OF SPEECH SEPARATION
](https://arxiv.org/pdf/2010.15366.pdf), which is not organized yet.

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
### Examples
- Conv-TasNet (default)
- WSJ0-2mix (default)
- training from scratch (default)
- multi-GPU, id=CUDA_VISIBLE_DEVICES (default)
```bash
bash ./run.sh --model ConvTasNet \
              --corpus wsj0-mix \
              --strategy from_scratch
```

- DPRNN
- Libri2Mix
- pretrain + fine-tune
- pretrained checkpoint: exp/xxx/model.pth
- single-GPU, id=0
```bash
bash ./run.sh --id 0 \
              --model DPRNNTasNet \
              --corpus LibriMix \
              --strategy pretrained \
              --load_path exp/xxx/model.pth
```

- DPTNet
- WSJ0-2mix (default)
- multi-task training
- speech denoise/enhancement data path: `data/librimix/wav8k/min/train-360/`
- multi-GPU, id=0,1,2,3
- total batch size: 8
- batch size on each gpu: 1 (need to accumulate gradient for 2 times per update)
```bash
bash ./run.sh --id 0,1,2,3 \
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
