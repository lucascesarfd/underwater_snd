import os
import torchaudio
import torch

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DeepShipDataset(Dataset):

    def __init__(self, metadata_file, target_sample_rate,
                 num_samples, transform=None, target_transform=None):
        self.metadata = self._get_metadata(metadata_file)
        self.transform = transform
        self.target_transform = target_transform
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.class_mapping = {'tug':0, 'tanker':1, 'cargo':2, 'passengership':3, 'background':4}
        #self.measures = ["t1_norm", "c1_norm", "p1_norm", "sal_norm", "sv_norm"]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        audio_sample_path = self.metadata.path.iloc[index]
        label = self._get_audio_sample_label(index)
        if self.target_transform:
            label = self.target_transform(label)

        signal, sr = torchaudio.load(
            audio_sample_path,
            frame_offset=self.metadata.sub_init.iloc[index],
            num_frames=self.num_samples,
        )
        signal = self._resample_to_target_sr(signal, sr)
        signal = self._mix_down_to_one_channel(signal)
        signal = self._cut_bigger_samples(signal)
        signal = self._right_pad_small_samples(signal)
        if self.transform:
            signal = self.transform(signal)

        #measures = self._get_physical_measures(index)

        return signal, label

    def _right_pad_small_samples(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_to_target_sr(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_to_one_channel(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_bigger_samples(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _get_audio_sample_label(self, index):
        label = self.metadata.label.iloc[index]
        return torch.tensor(self.class_mapping[label.lower()])

    def _get_metadata(self, metadata_file):
        metadata = pd.read_csv(metadata_file)
        return metadata
    
    #def _get_physical_measures(self, index):
    #    return self.metadata[self.measures].iloc[index].astype(np.float32).values


def create_data_loader(data, batch_size, shuffle=True):
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    return loader
