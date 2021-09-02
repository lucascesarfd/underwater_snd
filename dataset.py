import os
import torchaudio
import torch

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class DeepShipDataset(Dataset):

    def __init__(self, metadata_file, transformation,
                 target_sample_rate, num_samples):
        self.metadata = self._get_metadata(metadata_file)
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.class_mapping = {'tug':0, 'tanker':1, 'cargo':2, 'passengership':3, 'background':4}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        audio_sample_path = self.metadata.path.iloc[index]
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(
            audio_sample_path,
            frame_offset=self.metadata.sub_init.iloc[index],
            num_frames=self.num_samples,
        )
        signal = self._resample_to_target_sr(signal, sr)
        signal = self._mix_down_to_one_channel(signal)
        signal = self._cut_bigger_samples(signal)
        signal = self._right_pad_small_samples(signal)
        signal = self.transformation(signal)
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

    def _get_metadata(self, metadata_file, random_seed=42):
        metadata = pd.read_csv(metadata_file)
        metadata = metadata.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        return metadata


def create_data_loader(train_data, batch_size, validation_split=0.2, shuffle_dataset=False, random_seed=20):

    # Creating data indices for training and validation splits:
    dataset_size = len(train_data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, validation_loader