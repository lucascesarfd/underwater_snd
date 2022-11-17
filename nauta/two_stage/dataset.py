import torchaudio
import torch
import os

import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class VesselDeeperShipFeature(Dataset):
    """A class describing the output features from DeeperShip Dataset.
    """

    def __init__(self, root_path):
        self.files_list_mel = self._get_npy_list(os.path.join(root_path, "mel"))
        self.files_list_cqt = [x.replace("mel", "cqt") for x in self.files_list_mel]
        self.files_list_gammatone = [x.replace("mel", "gammatone") for x in self.files_list_mel]
        self.class_mapping = {'tug':0, 'tanker':1, 'cargo':2, 'passengership':3}

    def __len__(self):
        """Returns the lenght of the dataset.

        Returns:
            int: The lenght of the dataset.
        """
        return len(self.files_list_mel)

    def __getitem__(self, index):
        """Returns the item from the desired index.

        Args:
            index (int): The index of the desired data.

        Returns:
            tuple: The (signal,label) tuple
        """
        mel_feature_sample_path = os.path.normpath(self.files_list_mel[index])
        cqt_feature_sample_path = os.path.normpath(self.files_list_cqt[index])
        gammatone_feature_sample_path = os.path.normpath(self.files_list_gammatone[index])

        label = self._get_feature_sample_label(mel_feature_sample_path)

        mel_signal = self._get_feature_sample(mel_feature_sample_path)
        cqt_signal = self._get_feature_sample(cqt_feature_sample_path)
        gammatone_signal = self._get_feature_sample(gammatone_feature_sample_path)

        signal = torch.stack([mel_signal[0], cqt_signal, gammatone_signal])

        return signal, label

    def _get_npy_list(self, root_path):
        npy_list = []
        exclude = set(['background'])
        for root, dirs, files in os.walk(root_path, topdown=True):
            dirs[:] = [d for d in dirs if d not in exclude]
            for name in files:
                if name.endswith(".npy"):
                    npy_list.append(os.path.join(root, name))

        return npy_list

    def _get_feature_sample_label(self, feature_sample_path):
        label = os.path.basename(os.path.dirname(feature_sample_path))
        return torch.tensor(self.class_mapping[label.lower()])
    
    def _get_feature_sample(self, feature_sample_path):
        return torch.tensor(np.load(feature_sample_path))
