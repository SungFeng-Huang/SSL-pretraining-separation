from asteroid.losses import PITLossWrapper


class MultiTaskLossWrapper(PITLossWrapper):
    """ n_src separation + 1_src enhancement
    """
    def __init__(self, loss_func, pit_from="pw_mtx", perm_reduce=None):
        super().__init__(loss_func, pit_from=pit_from, perm_reduce=perm_reduce)

    def forward(self, est_targets, targets, **kwargs):
        n_src = targets.shape[1]
        if n_src == 1:
            # est_targets = est_targets[:, -1].reshape(est_targets.size(0), 1, est_targets.size(2))
            est_targets = est_targets[:, None, -1]
            return super().forward(est_targets, targets, **kwargs)
        else:
            assert est_targets.shape[1] == n_src + 1
            est_targets = est_targets[:, :-1]
            return super().forward(est_targets, targets, **kwargs)
