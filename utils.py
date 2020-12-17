import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from asteroid.data import LibriMix, WhamDataset, Wsj0mixDataset
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr


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
    if train_enh_dir:
        enh_infos = train_enh_set.get_infos()
        for key in enh_infos:
            infos["enh_"+key] = enh_infos[key]
    
    return train_loader, val_loader, infos

class MultiTaskLossWrapper(PITLossWrapper):
    """ n_src separation + 1_src enhancement
    """
    def __init__(self, loss_func, pit_from="pw_mtx", perm_reduce=None):
        super().__init__(loss_func, pit_from=pit_from, perm_reduce=perm_reduce)

    def forward(self, est_targets, targets, **kwargs):
        n_src = targets.shape[1]
        if n_src == 1:
            est_targets = est_targets[:, -1].reshape(est_targets.size(0), 1, est_targets.size(2))
            return super().forward(est_targets, targets, **kwargs)
        else:
            assert est_targets.shape[1] == n_src + 1
            est_targets = est_targets[:, :-1]
            return super().forward(est_targets, targets, **kwargs)


class MultiTaskBatchSampler(torch.utils.data.BatchSampler):

    def __init__(self, sampler, batch_size, drop_last, cum_thresholds):
        super().__init__(sampler, batch_size, drop_last)
        self.thresholds = cum_thresholds
        self.thres_ranges = list(zip(self.thresholds, self.thresholds[1:]))

    def __iter__(self):
        batches = [[] for _ in self.thres_ranges]
        for idx in self.sampler:
            for range_idx, (st, ed) in enumerate(self.thres_ranges):
                if st <= idx < ed:
                    batches[range_idx].append(idx)
                    if len(batches[range_idx]) == self.batch_size:
                        yield batches[range_idx]
                        batches[range_idx] = []
        for range_idx in range(len(self.thres_ranges)):
            if len(batches[range_idx]) > 0 and not self.drop_last:
                yield batches[range_idx]


def MultiTaskDataLoader(data_sources, shuffle, batch_size, drop_last, generator=None, **kwargs):
    # NOTE: would cause some errors when using pytorch_lightning ddp, since
    # pytorch_lightning would automatically use DistributedDataSampler.
    # Use pytorch_lightning dp mode instead.
    dataset = torch.utils.data.ConcatDataset(data_sources)
    cum_thresholds = [0]
    for data_source in data_sources:
        cum_thresholds.append(cum_thresholds[-1] + len(data_source))
    if shuffle:
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = MultiTaskBatchSampler(sampler, batch_size=batch_size, drop_last=drop_last, cum_thresholds=cum_thresholds)
    return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
