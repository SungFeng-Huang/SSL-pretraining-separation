from torch.utils.data import DataLoader
from asteroid.data import LibriMix, WhamDataset

from .utils import MultiTaskDataLoader


def make_dataloaders(corpus, train_dir, val_dir, train_enh_dir=None, task="sep_clean",
                     sample_rate=8000, n_src=2, segment=4.0, batch_size=4, num_workers=None,):
    if corpus == "LibriMix":
        train_set = LibriMix(csv_dir=train_dir, task=task, sample_rate=sample_rate, n_src=n_src, segment=segment,)
        val_set = LibriMix(csv_dir=val_dir, task=task, sample_rate=sample_rate, n_src=n_src, segment=segment,)
    elif corpus == "wsj0-mix":
        train_set = WhamDataset(json_dir=train_dir, task=task, sample_rate=sample_rate, nondefault_nsrc=n_src, segment=segment,)
        val_set = WhamDataset(json_dir=val_dir, task=task, sample_rate=sample_rate, nondefault_nsrc=n_src, segment=segment,)

    if train_enh_dir is None:
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True,)
    else:
        train_enh_set = LibriMix(csv_dir=train_enh_dir, task="enh_single", sample_rate=sample_rate, n_src=1, segment=segment,)
        train_loader = MultiTaskDataLoader([train_set, train_enh_set],
                                           shuffle=True, batch_size=batch_size, drop_last=True, num_workers=num_workers,)
    val_loader = DataLoader(val_set, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True,)
    
    infos = train_set.get_infos()
    # if train_enh_dir:
        # enh_infos = train_enh_set.get_infos()
        # for key in enh_infos:
            # infos["enh_"+key] = enh_infos[key]
    
    return train_loader, val_loader, infos


def make_test_dataset(corpus, test_dir, task="sep_clean", sample_rate=8000, n_src=2):
    if corpus == "LibriMix":
        test_set = LibriMix(csv_dir=test_dir, task=task, sample_rate=sample_rate, n_src=n_src, segment=None,)
    elif corpus == "wsj0-mix":
        test_set = WhamDataset(json_dir=test_dir, task=task, sample_rate=sample_rate, nondefault_nsrc=n_src, segment=None,)
    return test_set
