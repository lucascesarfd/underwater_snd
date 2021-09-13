import torch
import torchaudio

from torch import nn
from nnAudio import Spectrogram


def define_mel_spectrogram(sample_rate):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=64
    )
    return mel_spectrogram

def define_gamma_spectrogram(sample_rate):
    gamma_spectrogram = Spectrogram.Gammatonegram(
        sr=sample_rate, n_fft=1024, n_bins=64, hop_length=512,
        window='hann', center=True, pad_mode='reflect',
        power=2.0, htk=False, fmin=20.0, fmax=None, norm=1,
        trainable_bins=False, trainable_STFT=False, verbose=False
    )
    return gamma_spectrogram

def define_cqt_spectrogram(sample_rate):
    cqt_spectrogram = Spectrogram.CQT(
        sr=sample_rate, hop_length=512, fmin=32.7, fmax=None,
        n_bins=64, bins_per_octave=12, filter_scale=1, norm=1,
        window='hann', center=True, pad_mode='reflect', trainable=False,
        output_format='Magnitude', verbose=False
    )
    return cqt_spectrogram


pre_processing_layers = {
    "mel": define_mel_spectrogram,
    "gammatone": define_gamma_spectrogram,
    "cqt": define_cqt_spectrogram,
}


class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(64 * 63, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.flatten(input_data)
        logits = self.dense_layers(x)
        predictions = self.softmax(logits)
        return predictions


class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 5, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


models_list = {
    "feedforward": FeedForwardNet,
    "cnn": CNNNetwork,
}


def get_model(model_name="FeedForward", device="cpu"):
    model = models_list[model_name.lower()]().to(device)

    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)
    return model