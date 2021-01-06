import math
from math import ceil
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.activation import MultiheadAttention
from asteroid.masknn import activations, norms
from asteroid.utils import has_arg
from asteroid.dsp.overlap_add import DualPathProcessing


class PreLNTransformerLayer(nn.Module):
    """
    Pre-LN Transformer layer.

    Args:
        embed_dim (int): Number of input channels.
        n_heads (int): Number of attention heads.
        dim_ff (int): Number of neurons in the RNNs cell state.
            Defaults to 256. RNN here replaces standard FF linear layer in plain Transformer.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        activation (str, optional): activation function applied at the output of RNN.
        norm (str, optional): Type of normalization to use.

    References
        [1] Cem Subakan, Mirco Ravanelli, Samuele Cornell, Mirko Bronzi, and 
        Jianyuan Zhong. "Attention is All You Need in Speech Separation."
        arXiv (2020).
    """

    def __init__(
        self,
        embed_dim,
        n_heads,
        dim_ff,
        dropout=0.0,
        activation="relu",
        norm="gLN",
    ):
        super(PreLNTransformerLayer, self).__init__()

        self.mha = MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_dim, dim_ff)
        self.linear2 = nn.Linear(dim_ff, embed_dim)
        self.activation = activations.get(activation)()
        self.norm_mha = norms.get(norm)(embed_dim)
        self.norm_ff = norms.get(norm)(embed_dim)

    def forward(self, x):
        assert x.shape[0] != x.shape[1], "seq_len == channels would lead to wrong LN dimension"
        tomha = self.norm_mha(x)
        tomha = tomha.permute(2, 0, 1)
        # x is batch, channels, seq_len
        # mha is seq_len, batch, channels
        # self-attention is applied
        out = self.mha(tomha, tomha, tomha)[0]
        x = self.dropout(out.permute(1, 2, 0)) + x

        # lstm is applied
        toff = self.norm_ff(x)
        out = self.linear2(self.dropout(self.activation(self.linear1(toff.transpose(1, -1)))))
        x = self.dropout(out.transpose(1, -1)) + x
        return x

class SepFormerLayer(PreLNTransformerLayer):
    """
    SepFormer layer. Only the forward in different way.

    Args:
        embed_dim (int): Number of input channels.
        n_heads (int): Number of attention heads.
        dim_ff (int): Number of neurons in the RNNs cell state.
            Defaults to 256. RNN here replaces standard FF linear layer in plain Transformer.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        activation (str, optional): activation function applied at the output of RNN.
        norm (str, optional): Type of normalization to use.

    References
        [1] Cem Subakan, Mirco Ravanelli, Samuele Cornell, Mirko Bronzi, and 
        Jianyuan Zhong. "Attention is All You Need in Speech Separation."
        arXiv (2020).
    """
    def __init__(
        self,
        embed_dim,
        n_heads,
        dim_ff,
        dropout=0.0,
        activation="relu",
        norm="gLN",
    ):
        super().__init__(embed_dim, n_heads, dim_ff, dropout, activation, norm)

    def forward(self, x):
        assert x.shape[0] != x.shape[1], "seq_len == channels would lead to wrong LN dimension"
        tomha = self.norm_mha(x)
        tomha = tomha.permute(2, 0, 1)
        # x is batch, channels, seq_len
        # mha is seq_len, batch, channels
        # self-attention is applied
        out = self.mha(tomha, tomha, tomha)[0]
        x1 = self.dropout(out.permute(1, 2, 0)) + x

        # lstm is applied
        toff = self.norm_ff(x1)
        out = self.linear2(self.dropout(self.activation(self.linear1(toff.transpose(1, -1)))))
        x2 = self.dropout(out.transpose(1, -1)) + x
        return x2


class SepFormer(nn.Module):
    """SepFormer introduced in [1].

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        n_heads (int): Number of attention heads.
        ff_hid (int): Number of neurons in the RNNs cell state.
            Defaults to 256.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use.
        ff_activation (str, optional): activation function applied at the output of RNN.
        mask_act (str, optional): Which non-linear function to generate mask.
        dropout (float, optional): Dropout ratio, must be in [0,1].

    References
        [1] Cem Subakan, Mirco Ravanelli, Samuele Cornell, Mirko Bronzi, and 
        Jianyuan Zhong. "Attention is All You Need in Speech Separation."
        arXiv (2020).
    """

    def __init__(
        self,
        in_chan,
        n_src,
        n_heads=4,
        ff_hid=256,
        chunk_size=100,
        hop_size=None,
        n_repeats=2,
        k_repeats=4,
        norm_type="gLN",
        ff_activation="relu",
        mask_act="relu",
        dropout=0,
    ):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.n_heads = n_heads
        self.ff_hid = ff_hid
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.k_repeats = k_repeats
        self.n_src = n_src
        self.norm_type = norm_type
        self.ff_activation = ff_activation
        self.mask_act = mask_act
        self.dropout = dropout

        self.mha_in_dim = ceil(self.in_chan / self.n_heads) * self.n_heads
        if self.in_chan % self.n_heads != 0:
            warnings.warn(
                f"DPTransformer input dim ({self.in_chan}) is not a multiple of the number of "
                f"heads ({self.n_heads}). Adding extra linear layer at input to accomodate "
                f"(size [{self.in_chan} x {self.mha_in_dim}])"
            )
            self.input_layer = nn.Linear(self.in_chan, self.mha_in_dim)
        else:
            self.input_layer = None

        self.in_norm = norms.get(norm_type)(self.mha_in_dim)
        self.ola = DualPathProcessing(self.chunk_size, self.hop_size)

        # Succession of DPRNNBlocks.
        self.layers = nn.ModuleList([])
        for x in range(self.n_repeats):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Sequential(*[
                            PositionalEncoding(
                                self.mha_in_dim,
                                self.dropout
                            ),
                            *[
                                SepFormerLayer(
                                    self.mha_in_dim,
                                    self.n_heads,
                                    self.ff_hid,
                                    self.dropout,
                                    self.ff_activation,
                                    self.norm_type,
                                ) for _ in range(self.k_repeats)
                            ]
                        ]),
                        nn.Sequential(*[
                            PositionalEncoding(
                                self.mha_in_dim,
                                self.dropout
                            ),
                            *[
                                SepFormerLayer(
                                    self.mha_in_dim,
                                    self.n_heads,
                                    self.ff_hid,
                                    self.dropout,
                                    self.ff_activation,
                                    self.norm_type,
                                ) for _ in range(self.k_repeats)
                            ]
                        ]),
                    ]
                )
            )
        net_out_conv = nn.Conv2d(self.mha_in_dim, n_src * self.in_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        # Gating and masking in 2D space (after fold)
        self.mask_net = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1),
                                      nn.ReLU(),
                                      nn.Conv1d(self.in_chan, self.in_chan, 1))

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        r"""Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        if self.input_layer is not None:
            mixture_w = self.input_layer(mixture_w.transpose(1, 2)).transpose(1, 2)
        mixture_w = self.in_norm(mixture_w)  # [batch, bn_chan, n_frames]
        n_orig_frames = mixture_w.shape[-1]

        mixture_w = self.ola.unfold(mixture_w)
        batch, n_filters, self.chunk_size, n_chunks = mixture_w.size()

        for layer_idx in range(len(self.layers)):
            intra, inter = self.layers[layer_idx]
            mixture_w = self.ola.intra_process(mixture_w, intra)
            mixture_w = self.ola.inter_process(mixture_w, inter)

        output = self.first_out(mixture_w)
        output = output.reshape(batch * self.n_src, self.in_chan, self.chunk_size, n_chunks)
        output = self.ola.fold(output, output_size=n_orig_frames)

        output = self.mask_net(output)
        # Compute mask
        output = output.reshape(batch, self.n_src, self.in_chan, -1)
        est_mask = self.output_act(output)
        return est_mask

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "ff_hid": self.ff_hid,
            "n_heads": self.n_heads,
            "chunk_size": self.chunk_size,
            "hop_size": self.hop_size,
            "n_repeats": self.n_repeats,
            "k_repeats": self.k_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "ff_activation": self.ff_activation,
            "mask_act": self.mask_act,
            "dropout": self.dropout,
        }
        return config


class SepFormer2(nn.Module):
    """Modified SepFormer introduced in [1].

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        n_heads (int): Number of attention heads.
        ff_hid (int): Number of neurons in the RNNs cell state.
            Defaults to 256.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use.
        ff_activation (str, optional): activation function applied at the output of RNN.
        mask_act (str, optional): Which non-linear function to generate mask.
        dropout (float, optional): Dropout ratio, must be in [0,1].

    References
        [1] Cem Subakan, Mirco Ravanelli, Samuele Cornell, Mirko Bronzi, and 
        Jianyuan Zhong. "Attention is All You Need in Speech Separation."
        arXiv (2020).
    """

    def __init__(
        self,
        in_chan,
        n_src,
        n_heads=4,
        ff_hid=256,
        chunk_size=100,
        hop_size=None,
        n_repeats=2,
        k_repeats=4,
        norm_type="gLN",
        ff_activation="relu",
        mask_act="relu",
        dropout=0,
    ):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.n_heads = n_heads
        self.ff_hid = ff_hid
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.k_repeats = k_repeats
        self.n_src = n_src
        self.norm_type = norm_type
        self.ff_activation = ff_activation
        self.mask_act = mask_act
        self.dropout = dropout

        self.mha_in_dim = ceil(self.in_chan / self.n_heads) * self.n_heads
        if self.in_chan % self.n_heads != 0:
            warnings.warn(
                f"DPTransformer input dim ({self.in_chan}) is not a multiple of the number of "
                f"heads ({self.n_heads}). Adding extra linear layer at input to accomodate "
                f"(size [{self.in_chan} x {self.mha_in_dim}])"
            )
            self.input_layer = nn.Linear(self.in_chan, self.mha_in_dim)
        else:
            self.input_layer = None

        self.in_norm = norms.get(norm_type)(self.mha_in_dim)
        self.ola = DualPathProcessing(self.chunk_size, self.hop_size)

        # Succession of DPRNNBlocks.
        self.layers = nn.ModuleList([])
        for x in range(self.n_repeats):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Sequential(*[
                            PositionalEncoding(
                                self.mha_in_dim,
                                self.dropout
                            ),
                            *[
                                PreLNTransformerLayer(
                                    self.mha_in_dim,
                                    self.n_heads,
                                    self.ff_hid,
                                    self.dropout,
                                    self.ff_activation,
                                    self.norm_type,
                                ) for _ in range(self.k_repeats)
                            ]
                        ]),
                        nn.Sequential(*[
                            PositionalEncoding(
                                self.mha_in_dim,
                                self.dropout
                            ),
                            *[
                                PreLNTransformerLayer(
                                    self.mha_in_dim,
                                    self.n_heads,
                                    self.ff_hid,
                                    self.dropout,
                                    self.ff_activation,
                                    self.norm_type,
                                ) for _ in range(self.k_repeats)
                            ]
                        ]),
                    ]
                )
            )
        net_out_conv = nn.Conv2d(self.mha_in_dim, n_src * self.in_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        # Gating and masking in 2D space (after fold)
        self.net_out = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Sigmoid())

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        r"""Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        if self.input_layer is not None:
            mixture_w = self.input_layer(mixture_w.transpose(1, 2)).transpose(1, 2)
        mixture_w = self.in_norm(mixture_w)  # [batch, bn_chan, n_frames]
        n_orig_frames = mixture_w.shape[-1]

        mixture_w = self.ola.unfold(mixture_w)
        batch, n_filters, self.chunk_size, n_chunks = mixture_w.size()

        for layer_idx in range(len(self.layers)):
            intra, inter = self.layers[layer_idx]
            mixture_w = self.ola.intra_process(mixture_w, intra)
            mixture_w = self.ola.inter_process(mixture_w, inter)

        output = self.first_out(mixture_w)
        output = output.reshape(batch * self.n_src, self.in_chan, self.chunk_size, n_chunks)
        output = self.ola.fold(output, output_size=n_orig_frames)

        output = self.net_out(output) * self.net_gate(output)
        # Compute mask
        output = output.reshape(batch, self.n_src, self.in_chan, -1)
        est_mask = self.output_act(output)
        return est_mask

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "ff_hid": self.ff_hid,
            "n_heads": self.n_heads,
            "chunk_size": self.chunk_size,
            "hop_size": self.hop_size,
            "n_repeats": self.n_repeats,
            "k_repeats": self.k_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "ff_activation": self.ff_activation,
            "mask_act": self.mask_act,
            "dropout": self.dropout,
        }
        return config


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1) # seq_len, batch, channels
        pe = pe.transpose(0, 1).unsqueeze(0) # batch, channels, seq_len
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is seq_len, batch, channels
        # x = x + self.pe[:x.size(0), :]

        # x is batch, channels, seq_len
        x = x + self.pe[:, :, :x.size(2)]
        return self.dropout(x)
