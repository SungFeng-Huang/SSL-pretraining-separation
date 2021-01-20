# SSL-pretraining-separation
This is the official repository of [SELF-SUPERVISED PRE-TRAINING REDUCES LABEL PERMUTATION INSTABILITY OF SPEECH SEPARATION
](https://arxiv.org/pdf/2010.15366.pdf).

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
Run `scripts/*.sh` to reproduce experiments in the paper.

### Models
* [x] ConvTasNet
* [x] DPRNNTasNet
* [x] DPTNet
* [x] SepFormerTasNet (my implementation of [SepFormer](https://arxiv.org/pdf/2010.13154.pdf))
* [x] SepFormer2TasNet (my modification of [SepFormer](https://arxiv.org/pdf/2010.13154.pdf))


------------------------------------
Reference
------------------------------------
The codes were adapted from
- [asteroid/egs/librimix/ConvTasNet/](https://github.com/asteroid-team/asteroid/tree/master/egs/librimix/ConvTasNet)
- [asteroid/egs/wham/ConvTasNet/](https://github.com/asteroid-team/asteroid/tree/master/egs/wham/ConvTasNet).
