#echo "exp/train_convtasnet_ConvTasNet_Libri2Mix_sep_clean/_ckpt_epoch_99.ckpt";
#python eval_general.py  --use_gpu 1 \
  #--corpus LibriMix --test_dir data/wav8k/min/test --task sep_clean \
  #--exp_dir exp/train_convtasnet_ConvTasNet_Libri2Mix_sep_clean --ckpt_path _ckpt_epoch_99.ckpt --out_dir eval_99_tt;

#echo "exp/train_convtasnet_ConvTasNet_Libri2Mix_sep_clean_pretrained/_ckpt_epoch_98.ckpt";
#python eval_general.py  --use_gpu 1 \
  #--corpus LibriMix --test_dir data/wav8k/min/test --task sep_clean \
  #--exp_dir exp/train_convtasnet_ConvTasNet_Libri2Mix_sep_clean_pretrained --ckpt_path _ckpt_epoch_98.ckpt --out_dir eval_98_tt;

#echo "exp/train_convtasnet_ConvTasNet_Libri2Mix_sep_clean_multi_task_enh360/_ckpt_epoch_97.ckpt";
#python eval_general.py  --use_gpu 1 \
  #--corpus LibriMix --test_dir data/wav8k/min/test --task sep_clean \
  #--exp_dir exp/train_convtasnet_ConvTasNet_Libri2Mix_sep_clean_multi_task_enh360 --ckpt_path _ckpt_epoch_97.ckpt --out_dir eval_97_tt;

#echo "exp/train_convtasnet_ConvTasNet_wsj0-2mix_sep_clean/_ckpt_epoch_98.ckpt";
#python eval_general.py --use_gpu 1\
  #--corpus wsj0-mix --test_dir wsj0-mix/2speakers/wav8k/min/tt --task sep_clean \
  #--exp_dir exp/train_convtasnet_ConvTasNet_wsj0-2mix_sep_clean --ckpt_path _ckpt_epoch_98.ckpt --out_dir eval_98_tt;

#echo "exp/train_convtasnet_ConvTasNet_wsj0-2mix_sep_clean_pretrained_old/_ckpt_epoch_99.ckpt";
#python eval_general.py --use_gpu 1\
  #--corpus wsj0-mix --test_dir wsj0-mix/2speakers/wav8k/min/tt --task sep_clean \
  #--exp_dir exp/train_convtasnet_ConvTasNet_wsj0-2mix_sep_clean_pretrained_old --ckpt_path _ckpt_epoch_99.ckpt --out_dir eval_99_tt;

#echo "exp/train_convtasnet_ConvTasNet_wsj0-2mix_sep_clean_multi_task_enh360/_ckpt_epoch_94.ckpt";
#python eval_general.py --use_gpu 1\
  #--corpus wsj0-mix --test_dir wsj0-mix/2speakers/wav8k/min/tt --task sep_clean \
  #--exp_dir exp/train_convtasnet_ConvTasNet_wsj0-2mix_sep_clean_multi_task_enh360 --ckpt_path _ckpt_epoch_94.ckpt --out_dir eval_94_tt;

#echo "exp/train_dprnn_DPRNNTasNet_wsj0-2mix_sep_clean_new_8gpu/_ckpt_epoch_171.ckpt";
#python eval_general.py --use_gpu 1 --model DPRNNTasNet \
  #--corpus wsj0-mix --test_dir wsj0-mix/2speakers/wav8k/min/tt --task sep_clean \
  #--exp_dir exp/train_dprnn_DPRNNTasNet_wsj0-2mix_sep_clean_new_8gpu --ckpt_path _ckpt_epoch_171.ckpt --out_dir eval_171_tt;

#echo "exp/train_dprnn_DPRNNTasNet_wsj0-2mix_sep_clean_new_8gpu_pretrained/_ckpt_epoch_122.ckpt";
#python eval_general.py --use_gpu 1 --model DPRNNTasNet \
  #--corpus wsj0-mix --test_dir wsj0-mix/2speakers/wav8k/min/tt --task sep_clean \
  #--exp_dir exp/train_dprnn_DPRNNTasNet_wsj0-2mix_sep_clean_new_8gpu_pretrained --ckpt_path _ckpt_epoch_122.ckpt --out_dir eval_122_tt;

#echo "exp/train_dptnet_DPTNet_wsj0-2mix_sep_clean/_ckpt_epoch_196.ckpt";
#python eval_general.py --use_gpu 1 --model DPTNet \
  #--corpus wsj0-mix --test_dir wsj0-mix/2speakers/wav8k/min/tt --task sep_clean \
  #--exp_dir exp/train_dptnet_DPTNet_wsj0-2mix_sep_clean --ckpt_path _ckpt_epoch_196.ckpt --out_dir eval_196_tt;
#exit

echo "exp/train_dptnet_DPTNet_wsj0-2mix_sep_clean_8gpu_pretrained/_ckpt_epoch_199.ckpt";
python eval_general.py --use_gpu 1 --model DPTNet \
  --corpus wsj0-mix --test_dir wsj0-mix/2speakers/wav8k/min/tt --task sep_clean \
  --exp_dir exp/train_dptnet_DPTNet_wsj0-2mix_sep_clean_8gpu_pretrained --ckpt_path _ckpt_epoch_199.ckpt --out_dir eval_199_tt;
