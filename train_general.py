import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import asteroid
from asteroid.models import ConvTasNet, DPRNNTasNet, DPTNet
from asteroid.data import LibriMix
from asteroid.data.wsj0_mix import Wsj0mixDataset
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.engine.schedulers import DPTNetScheduler
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

from utils import MultiTaskDataLoader, MultiTaskLossWrapper, MultiTaskBatchSampler

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--corpus", default="LibriMix", help="LibriMix / wsj0-mix")
parser.add_argument("--model", default="ConvTasNet", help="ConvTasNet / DPRNNTasNet / DPTNet")
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
parser.add_argument("--load_path", default=None, help="Checkpoint path to load.")
parser.add_argument("--real_batch_size", type=int, default=12, help="Using accumulate_grad_batches.")
parser.add_argument("--multi_task", action="store_true", help="Multi-task training with speech enhancement.")
parser.add_argument("--train_enh_dir", default=None, help="Multi-task data dir.")
parser.add_argument("--resume_ckpt", default=None, help="resume training")

# MODEL = {
    # "ConvTasNet": ConvTasNet,
    # "DPRNNTasNet": DPRNNTasNet,
    # "DPTNet": DPTNet,
# }


def main(conf):
    if conf["main_args"]["corpus"] == "LibriMix":
        train_set = LibriMix(
            csv_dir=conf["data"]["train_dir"],
            task=conf["data"]["task"],
            sample_rate=conf["data"]["sample_rate"],
            n_src=conf["data"]["n_src"],
            segment=conf["data"]["segment"],
        )

        val_set = LibriMix(
            csv_dir=conf["data"]["valid_dir"],
            task=conf["data"]["task"],
            sample_rate=conf["data"]["sample_rate"],
            n_src=conf["data"]["n_src"],
            segment=conf["data"]["segment"],
        )
    elif conf["main_args"]["corpus"] == "wsj0-mix":
        train_set = Wsj0mixDataset(
            json_dir=conf["data"]["train_dir"],
            sample_rate=conf["data"]["sample_rate"],
            n_src=conf["data"]["n_src"],
            segment=conf["data"]["segment"],
        )

        val_set = Wsj0mixDataset(
            json_dir=conf["data"]["valid_dir"],
            sample_rate=conf["data"]["sample_rate"],
            n_src=conf["data"]["n_src"],
            segment=conf["data"]["segment"],
        )

    if not conf["main_args"]["multi_task"]:
        train_loader = DataLoader(
            train_set,
            shuffle=True,
            batch_size=conf["main_args"]["real_batch_size"],
            drop_last=True,
            num_workers=conf["training"]["num_workers"],
        )

        val_loader = DataLoader(
            val_set,
            shuffle=False,
            batch_size=conf["main_args"]["real_batch_size"],
            num_workers=conf["training"]["num_workers"],
            drop_last=True,
        )
        conf["masknet"].update({"n_src": conf["data"]["n_src"]})

    else:
        train_enh_set = LibriMix(
            csv_dir=conf["main_args"]["train_enh_dir"],
            task="enh_single",
            sample_rate=conf["data"]["sample_rate"],
            n_src=1,
            segment=conf["data"]["segment"],
        )

        # val_enh_set = LibriMix(
            # csv_dir=conf["main_args"]["valid_enh_dir"],
            # task="enh_single",
            # sample_rate=conf["data"]["sample_rate"],
            # n_src=1,
            # segment=conf["data"]["segment"],
        # )

        train_loader = MultiTaskDataLoader(
            [train_set, train_enh_set],
            shuffle=True,
            batch_size=conf["main_args"]["real_batch_size"],
            drop_last=True,
            num_workers=conf["training"]["num_workers"],
        )

        val_loader = DataLoader(
            val_set,
            shuffle=False,
            batch_size=conf["main_args"]["real_batch_size"],
            num_workers=conf["training"]["num_workers"],
            drop_last=True,
        )

        # val_enh_loader = DataLoader(
            # val_enh_set,
            # shuffle=False,
            # batch_size=conf["main_args"]["real_batch_size"],
            # num_workers=conf["training"]["num_workers"],
            # drop_last=True,
        # )
        conf["masknet"].update({"n_src": conf["data"]["n_src"] + 1})

    model = getattr(asteroid, conf["main_args"]["model"])(**conf["filterbank"], **conf["masknet"])
    # model = ConvTasNet(**conf["filterbank"], **conf["masknet"])
    if conf["main_args"]["load_path"] is not None:
        all_states = torch.load(conf["main_args"]["load_path"], map_location="cpu")
        if "state_dict" in all_states:
            del all_states["state_dict"]["masker.mask_net.1.weight"]
            del all_states["state_dict"]["masker.mask_net.1.bias"]
            model.load_state_dict(all_states["state_dict"], strict=False)
        else:
            del all_states["model"]["masker.mask_net.1.weight"]
            del all_states["model"]["masker.mask_net.1.bias"]
            model.load_state_dict(all_states["model"], strict=False)
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["main_args"]["model"] == "DPTNet":
        scheduler = {
            "scheduler": DPTNetScheduler(
                optimizer, len(train_loader) // conf["training"]["batch_size"], 64
            ),
            "interval": "batch",
        }
    elif conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    if conf["main_args"]["multi_task"]:
        loss_func = MultiTaskLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    else:
        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        # val_loader=[val_loader, val_enh_loader],
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
        # checkpoint_dir, monitor="val_loss", mode="min", save_top_k=-1, verbose=True
    )
    early_stopping = False
    if conf["training"]["early_stop"]:
        early_stopping = EarlyStopping(monitor="val_loss", patience=30, verbose=True)


    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stopping,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend="dp",
        train_percent_check=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        accumulate_grad_batches=int(conf["training"]["batch_size"]/conf["main_args"]["real_batch_size"]),
        resume_from_checkpoint=conf["main_args"]["resume_ckpt"],
    )
    trainer.fit(system)
    # trainer.fit(system, val_dataloaders=[val_loader, val_enh_loader])

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    # state_dict = torch.load(checkpoint.best_model_path)
    # system.load_state_dict(state_dict=state_dict["state_dict"])
    # system.cpu()

    # to_save = system.model.serialize()
    # to_save.update(train_set.get_infos())
    # torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint as print
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    print(arg_dic)
    main(arg_dic)
