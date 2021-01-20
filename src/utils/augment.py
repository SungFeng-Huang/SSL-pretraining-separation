from functools import partial
import torch
from torch import nn
import os
import glob


class Chopper(nn.Module):
    def __init__(self, chop_factors=[(0.05, 0.025), (0.1, 0.05)],
                 max_chops=2, force_regions=True):
        # chop factors in seconds (mean, std) per possible chop
        import webrtcvad
        self.chop_factors = chop_factors
        self.max_chops = max_chops
        self.force_regions = force_regions
        # make scalers to norm/denorm
        self.denormalizer = Scale(1. / ((2 ** 15) - 1))
        self.normalizer = Scale((2 ** 15) - 1)

    def vad_wav(self, wav, srate):
        """ Detect the voice activity in the 16-bit mono PCM wav and return
            a list of tuples: (speech_region_i_beg_sample, center_sample,
            region_duration)
        """
        if srate != 16000:
            raise ValueError('Sample rate must be 16kHz')
        window_size = 160  # samples
        regions = []
        curr_region_counter = 0
        init = None
        vad = self.vad
        if self.force_regions:
            # Divide the signal into even regions depending on number of chops
            # to put
            nregions = wav.shape[0] // self.max_chops
            reg_len = wav.shape[0] // nregions
            for beg_i in range(0, wav.shape[0], reg_len):
                end_sample = beg_i + reg_len
                center_sample = beg_i + (end_sample - beg_i) / 2
                regions.append((beg_i, center_sample,
                                reg_len))
            return regions


class BandDrop(nn.Module):
    def __init__(self, filt_files, filt_fmt='npy', data_root='.'):
        if len(filt_files) == 0:
            filt_files = [os.path.basename(f) for f in glob.glob(os.path.join(data_root, '*.{}'.format(filt_fmt)))]
        print('Found {} *.{} filt_files in {}'.format(len(filt_files), filt_fmt, data_root))

        self.filt_files = filt_files
        assert isinstance(filt_files, list), type(filt_files)
        assert len(filt_files) > 0, len(filt_files)
        self.filt_idxs = list(range(len(filt_files)))
        self.filt_fmt = filt_fmt
        self.data_root = data_root

    def load_filter(self, filt_file, filt_fmt):

        filt_file = os.path.join(self.data_root, filt_file)

        if filt_fmt == 'mat':
            filt_coeff = loadmat(filt_file, squeeze_me=True, struct_as_record=False)
            filt_coeff = filt_coeff['filt_coeff']
        elif filt_fmt == 'imp' or filt_fmt == 'txt':
            filt_coeff = np.loadtxt(filt_file)
        elif filt_fmt == 'npy':
            filt_coeff = np.load(filt_file)
        else:
            raise TypeError('Unrecognized filter format: ', filt_fmt)

        filt_coeff = filt_coeff / np.abs(np.max(filt_coeff))

        return filt_coeff

    def shift(self, xs, n):
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = 0.0
            e[n:] = xs[:-n]
        else:
            e[n:] = 0.0
            e[:n] = xs[-n:]
        return e

    def sample_filt(self):
        #TODO: maybe use data loader for prefetch
        if len(self.filt_files) == 0:
            return self.filt_files[0]
        else:
            idx = random.choice(self.filt_idxs)
            return self.filt_files[idx]

    def forward(self, wav):
        # sample a filter
        filt_file = self.sample_filt()
        filt_coeff = self.load_filter(filt_file, self.filt_fmt)
        filt_coeff = filt_coeff.astype(np.float32)
        wav = wav.data.numpy().reshape(-1)
        Ex = np.dot(wav, wav)
        wav = wav.astype(np.float32).reshape(-1)

        #TODO: pytorch version for cuda
        sig_filt = signal.convolve(wav, filt_coeff, mode='full').reshape(-1)
        # sig_filt = torch.nn.functional.conv1d(
            # wav[None, None, :],
            # torch.flip(torch.from_numpy(filt_coeff), (0,))[None, None, :], padding=len(filt_coeff)-1
        # ).reshape(-1)

        sig_filt = self.shift(sig_filt, -round(filt_coeff.shape[0] / 2))

        sig_filt = sig_filt[:wav.shape[0]]

        Efilt = np.dot(sig_filt, sig_filt)

        if Efilt > 0:
            Eratio = np.sqrt(Ex / Efilt)
        else:
            Eratio = 1.0
            sig_filt = wav

        sig_filt = Eratio * sig_filt
        sig_filt = torch.FloatTensor(sig_filt)
        return sig_filt

    def __repr__(self):
        if len(self.filt_files) > 3:
            attrs = '(filt_files={} ...)'.format(self.filt_files[:3])
        else:
            attrs = '(filt_files={})'.format(self.filt_files)
        return self.__class__.__name__ + attrs




# from pase.transforms import BandDrop, Chopper

# trans.append(Chopper(max_chops=max_chops,
                     # chop_factors=chop_factors,
                     # report=report))
# probs.append(chop_p)

# trans.append(BandDrop(bandrop_irfiles,filt_fmt=bandrop_fmt,
                      # data_root=bandrop_data_root,
                      # report=report))
# probs.append(bandrop_p)

