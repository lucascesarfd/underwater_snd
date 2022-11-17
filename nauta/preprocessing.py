from nnAudio import Spectrogram

FREQ_BINS = 95 # This number was based on the CQT, which have 95 freq bins for 4186hz
HOP_LENGTH = 256 # Used to generate an output of 128 on x axis
N_FFT = 2048 # This value is UNUSED because of the freq bins is mandatory
FMAX = 4186 # Correspond to a C8 note (Most High on a piano) (empirical)
FMIN = 18.0 # Minimun accepted value on CQT for audios of 1s

def define_mel_spectrogram(sample_rate):
    """Returns a MelSpectrogram transforms object.

    Args:
        sample_rate (int): The desired sample rate.

    Returns:
        torchaudio.transforms: The MelSpectrogram object initialized.
    """
    mel_spectrogram = Spectrogram.MelSpectrogram(
        sr=sample_rate, n_fft=N_FFT, n_mels=FREQ_BINS, hop_length=HOP_LENGTH,
        window='hann', center=True, pad_mode='reflect',
        power=2.0, htk=False, fmin=FMIN, fmax=FMAX, norm=1,
        trainable_mel=False, trainable_STFT=False, verbose=False
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
        sr=sample_rate, n_fft=N_FFT, n_bins=FREQ_BINS, hop_length=HOP_LENGTH,
        window='hann', center=True, pad_mode='reflect',
        power=2.0, htk=False, fmin=FMIN, fmax=FMAX, norm=1,
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
        sr=sample_rate, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX,
        n_bins=FREQ_BINS, bins_per_octave=12, filter_scale=1, norm=1,
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
