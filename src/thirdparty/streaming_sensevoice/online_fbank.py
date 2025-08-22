import kaldi_native_fbank as knf
import numpy as np


class OnlineFbank(knf.OnlineFbank):
    def __init__(self, sample_rate=16000, window_type="povey", dither=0, num_bins=80):
        opts = knf.FbankOptions()
        opts.frame_opts.dither = dither
        opts.frame_opts.samp_freq = sample_rate
        opts.frame_opts.snip_edges = False
        opts.frame_opts.window_type = window_type
        opts.mel_opts.debug_mel = False
        opts.mel_opts.num_bins = num_bins
        super().__init__(opts)

        self.sample_rate = sample_rate
        self.first_available_index = 0

    def accept_waveform(self, samples, is_last=False):
        super().accept_waveform(self.sample_rate, samples)
        if is_last:
            super().input_finished()

    def num_frames_ready(self):
        return super().num_frames_ready - self.first_available_index

    def get_frame(self, index):
        return super().get_frame(index + self.first_available_index)

    def get_frames(self, num_frames=None):
        num_frames = num_frames or self.num_frames_ready()
        frames = []
        for i in range(num_frames):
            frames.append(self.get_frame(i))
        self.first_available_index += num_frames
        return np.stack(frames) if len(frames) > 0 else None

    def get_lfr_frames(self, window_size=7, window_shift=6, neg_mean=0, inv_stddev=1):
        num_lfr_frames = (self.num_frames_ready() - window_size) // window_shift + 1
        if num_lfr_frames <= 0:
            return np.array([])

        num_frames = window_size + (num_lfr_frames - 1) * window_shift
        frames = self.get_frames(num_frames)
        self.first_available_index -= window_size - window_shift
        lfr_frames = np.lib.stride_tricks.as_strided(
            frames,
            shape=(num_lfr_frames, frames.shape[1] * window_size),
            strides=((window_shift * frames.shape[1]) * 4, 4),
        )
        return (lfr_frames + neg_mean) * inv_stddev
