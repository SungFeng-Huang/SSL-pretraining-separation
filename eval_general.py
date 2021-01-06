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
parser.add_argument("--out_dir", type=str, default="results/best_model", help="Directory in exp_dir where the eval results will be stored")
parser.add_argument("--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all")

parser.add_argument("--ckpt_path", default="best_model.pth", help="Experiment checkpoint path")
parser.add_argument("--publishable", action="store_true", help="Save publishable.")

compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(conf):
    model_path = os.path.join(conf["exp_dir"], conf["ckpt_path"])

    # all resulting files would be saved in eval_save_dir
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    os.makedirs(eval_save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(eval_save_dir, "final_metrics.json")):
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

        # Randomly choose the indexes of sentences to save.
        ex_save_dir = os.path.join(eval_save_dir, "examples/")
        if conf["n_save_ex"] == -1:
            conf["n_save_ex"] = len(test_set)
        save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])

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

            loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
            mix_np = mix.cpu().data.numpy()
            sources_np = sources.cpu().data.numpy()
            est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
            # For each utterance, we get a dictionary with the mixture path,
            # the input and output metrics
            utt_metrics = get_metrics(
                mix_np,
                sources_np,
                est_sources_np,
                sample_rate=conf["sample_rate"],
                metrics_list=compute_metrics,
            )
            if hasattr(test_set, "mixture_path"):
                utt_metrics["mix_path"] = test_set.mixture_path
            series_list.append(pd.Series(utt_metrics))

            # Save some examples in a folder. Wav files and metrics as text.
            if idx in save_idx:
                local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
                os.makedirs(local_save_dir, exist_ok=True)
                sf.write(local_save_dir + "mixture.wav", mix_np, conf["sample_rate"])
                # Loop over the sources and estimates
                for src_idx, src in enumerate(sources_np):
                    sf.write(local_save_dir + "s{}.wav".format(src_idx), src, conf["sample_rate"])
                for src_idx, est_src in enumerate(est_sources_np):
                    est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
                    sf.write(
                        local_save_dir + "s{}_estimate.wav".format(src_idx),
                        est_src,
                        conf["sample_rate"],
                    )
                # Write local metrics to the example folder.
                with open(local_save_dir + "metrics.json", "w") as f:
                    json.dump(utt_metrics, f, indent=0)

        # Save all metrics to the experiment folder.
        all_metrics_df = pd.DataFrame(series_list)
        all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

        # Print and save summary metrics
        final_results = {}
        for metric_name in compute_metrics:
            input_metric_name = "input_" + metric_name
            ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
            final_results[metric_name] = all_metrics_df[metric_name].mean()
            final_results[metric_name + "_imp"] = ldf.mean()
        print("Overall metrics :")
        pprint(final_results)
        with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
            json.dump(final_results, f, indent=0)
    else:
        with open(os.path.join(eval_save_dir, "final_metrics.json"), "r") as f:
            final_results = json.load(f)

    if conf["publishable"]:
        assert conf["ckpt_path"] == "best_model.pth"
        model_dict = torch.load(model_path, map_location="cpu")
        os.makedirs(os.path.join(conf["exp_dir"], "publish_dir"), exist_ok=True)
        publishable = save_publishable(
            os.path.join(conf["exp_dir"], "publish_dir"),
            model_dict,
            metrics=final_results,
            train_conf=train_conf,
        )


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
