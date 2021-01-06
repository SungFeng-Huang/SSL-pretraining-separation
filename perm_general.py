import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from itertools import permutations

import asteroid
from asteroid.metrics import get_metrics
from asteroid.data.librimix_dataset import LibriMix
from asteroid.data.wsj0_mix import Wsj0mixDataset
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device

from utils import make_test_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--corpus", default="LibriMix", choices=["LibriMix", "wsj0-mix"])
parser.add_argument("--model", default="ConvTasNet", choices=["ConvTasNet", "DPRNNTasNet", "DPTNet"])
parser.add_argument("--test_dir", type=str, required=True, help="Test directory including the csv files")
parser.add_argument("--task", type=str, default="sep_clean", choices=["sep_clean", "sep_noisy"])
parser.add_argument("--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution")
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument("--out_dir", type=str, required=True, help="Directory in exp_dir where the eval results will be stored")

parser.add_argument("--ckpt_path", default="best_model.pth", help="Experiment checkpoint path")

compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(conf):
    perms = list(permutations(range(conf["train_conf"]["data"]["n_src"])))

    model_path = os.path.join(conf["exp_dir"], conf["ckpt_path"])
    if conf["ckpt_path"] == "best_model.pth":
        # serialized checkpoint
        model = getattr(asteroid, conf["model"]).from_pretrained(model_path)
    else:
        # non-serialized checkpoint, _ckpt_epoch_{i}.ckpt, keys would start with
        # "model.", which need to be removed
        model = getattr(asteroid, conf["model"])(**conf["train_conf"]["filterbank"], **conf["train_conf"]["masknet"])
        all_states = torch.load(model_path, map_location="cpu")
        state_dict = {k.split('.', 1)[1]: all_states["state_dict"][k] for k in all_states["state_dict"]}
        model.load_state_dict(state_dict)
        # model.load_state_dict(all_states["state_dict"], strict=False)

    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = make_test_dataset(
        corpus=conf["corpus"], 
        test_dir=conf["test_dir"],
        task=conf["task"],
        sample_rate=conf["sample_rate"],
        n_src=conf["train_conf"]["data"]["n_src"],
        )
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # all resulting files would be saved in eval_save_dir
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    os.makedirs(eval_save_dir, exist_ok=True)

    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources = tensors_to_device(test_set[idx], device=model_device)
        est_sources = model(mix.unsqueeze(0))

        # When inferencing separation for multi-task training,
        # exclude the last channel. Does not effect single-task training
        # models (from_scratch, pre+FT).
        est_sources = est_sources[:, :sources.shape[0]]
        _, best_perm_idx = loss_func.find_best_perm(pairwise_neg_sisdr(est_sources, sources[None]), conf["train_conf"]["data"]["n_src"])

        utt_metrics = {}
        if hasattr(test_set, "mixture_path"):
            utt_metrics["mix_path"] = test_set.mixture_path
        utt_metrics["best_perm_idx"] = ' '.join([str(pidx) for pidx in perms[best_perm_idx[0]]])
        series_list.append(pd.Series(utt_metrics))

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "best_perms.csv"))


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    if args.task != arg_dic["train_conf"]["data"]["task"]:
        print(
            "Warning : the task used to test is different than "
            "the one from training, be sure this is what you want."
        )

    main(arg_dic)
