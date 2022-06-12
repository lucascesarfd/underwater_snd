import torchaudio
from nnAudio import Spectrogram

def define_mel_spectrogram(sample_rate):
    """Returns a MelSpectrogram transforms object.

    Args:
        sample_rate (int): The desired sample rate.

    Returns:
        torchaudio.transforms: The MelSpectrogram object initialized.
    """
    #mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    #    sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=64
    #)
    mel_spectrogram = Spectrogram.MelSpectrogram(
        sr=sample_rate, n_fft=1024, n_mels=64, hop_length=512,
        window='hann', center=True, pad_mode='reflect', power=2.0,
        htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False,
        trainable_STFT=False, verbose=False
    )
    return mel_spectrogram


def define_gamma_spectrogram(sample_rate):
    """Returns a Gammatonegram object.

    Args:
        sample_rate (int): The desired sample rate.

    Returns:
        Spectrogram: The Gammatonegram object initialized.
    """
    gamma_spectrogram = Spectrogram.Gammatonegram(
        sr=sample_rate, n_fft=1024, n_bins=64, hop_length=512,
        window='hann', center=True, pad_mode='reflect',
        power=2.0, htk=False, fmin=20.0, fmax=None, norm=1,
        trainable_bins=False, trainable_STFT=False, verbose=False
    )
    return gamma_spectrogram


def define_cqt_spectrogram(sample_rate):
    """Returns a CQT object.

    Args:
        sample_rate (int): The desired sample rate.

    Returns:
        Spectrogram: The CQT object initialized.
    """
    cqt_spectrogram = Spectrogram.CQT(
        sr=sample_rate, hop_length=512, fmin=32.7, fmax=None,
        n_bins=64, bins_per_octave=12, filter_scale=1, norm=1,
        window='hann', center=True, pad_mode='reflect', trainable=False,
        output_format='Magnitude', verbose=False
    )
    return cqt_spectrogram


_pre_processing_layers = {
    "mel": define_mel_spectrogram,
    "gammatone": define_gamma_spectrogram,
    "cqt": define_cqt_spectrogram,
}

def get_preprocessing_layer(pre_processing_type, sample_rate):
    return _pre_processing_layers[pre_processing_type](sample_rate)
