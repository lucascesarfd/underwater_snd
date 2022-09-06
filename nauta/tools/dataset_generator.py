"""
This file is used to generate the preprocessed version of the dataset.

It receives the metadata CSV file and saves the CQT, Gammatone and Mel preprocessed
versions of the dataset, on npy files.

Is also possible to save the audio chunk files setting the flag saveaudio to True.
"""
import argparse
import os
import torch
import torchaudio
import yaml

import numpy as np
import pandas as pd

from tqdm import tqdm
from nauta.tools.utils import create_dir
from nauta.preprocessing import get_preprocessing_layer

def create_parser():
    """Create the parser object.

    Returns:
        parser: The generated parser object with arguments
    """
    parser = argparse.ArgumentParser(description="Execute the training routine.")

    parser.add_argument(
        "config_file",
        type=str,
        help="",
    )

    return parser

def right_pad_small_samples(signal, num_samples):
    """Insert a pad at the right side of the data

    Args:
        signal (tensor): The input signal.

    Returns:
        tensor: The processed signal.
    """
    length_signal = signal.shape[1]
    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal

def resample_to_target_sr(signal, sr, target_sample_rate):
    """Resample audio to desired sample rate.

    Args:
        signal (tensor): The input signal.
        sr (int): The desired sample rate.

    Returns:
        tensor: The processed signal.
    """
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)
    return signal

def mix_down_to_one_channel(signal):
    """Unify the data into ione channel.

    Args:
        signal (tensor): The input signal.

    Returns:
        tensor: The processed signal.
    """
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

def cut_bigger_samples(signal, num_samples):
    """Cut the signal to the desired num of samples.

    Args:
        signal (tensor): The input signal.

    Returns:
        tensor: The processed signal.
    """
    if signal.shape[1] > num_samples:
        signal = signal[:, : num_samples]
    return signal

def get_full_audio(audio_sample_path, target_sample_rate):
    signal, sr = torchaudio.load(audio_sample_path)

    signal = resample_to_target_sr(signal, sr, target_sample_rate)
    signal = mix_down_to_one_channel(signal)
    return signal

def get_audio_chunk(signal, seconds_init, sample_rate, num_samples):
    initial_offset = int(seconds_init*sample_rate)
    final_offset = initial_offset + num_samples
    signal_chunk = signal[:, initial_offset:final_offset]
    signal_chunk = cut_bigger_samples(signal_chunk, num_samples)
    signal_chunk = right_pad_small_samples(signal_chunk, num_samples)
    return signal_chunk

def get_interleaved_metadata(metadata_path):
    original_metadata = pd.read_csv(metadata_path)
    interleaved_metadata = original_metadata.copy()
    interleaved_metadata["sub_init"] = interleaved_metadata["sub_init"].apply(lambda row: row + 0.5)
    metadata = pd.concat([original_metadata, interleaved_metadata])
    metadata = metadata.reset_index(drop=True)
    return metadata

def generate_dataset_artifacts(metadata_path, output_dir, target_sample_rate, number_of_samples, interleaved=True, save_audio=True):
    artifacts_list = ["mel", "cqt", "gammatone"]
    if save_audio:
        artifacts_list.append("audio")

    if interleaved:
        metadata = get_interleaved_metadata(metadata_path)
    else:
        metadata = pd.read_csv(metadata_path)

    metadata['file_index'] = metadata.index

    labels_names = list(metadata["label"].unique())
    for preprocessing in artifacts_list:
        create_dir(os.path.join(output_dir, preprocessing))
        for label in labels_names:
            create_dir(os.path.join(output_dir, preprocessing, label))

    transformation_mel = get_preprocessing_layer("mel", target_sample_rate)
    transformation_cqt = get_preprocessing_layer("cqt", target_sample_rate)
    transformation_gamma = get_preprocessing_layer("gammatone", target_sample_rate)

    metadata_group = metadata.groupby(["label", "path", "sample_rate"])
    for (label, path, sample_rate), df in tqdm(metadata_group):
        audio_signal = get_full_audio(path, target_sample_rate)
        for _, row in df.iterrows():
            sec_init = row["sub_init"]
            idx = row["file_index"]
            audio_chunk = get_audio_chunk(audio_signal, sec_init, target_sample_rate, number_of_samples)

            # Audio
            if "audio" in artifacts_list:
                audio_path = os.path.join(output_dir, "audio", label, f"{idx}.wav")
                torchaudio.save(audio_path, audio_chunk, target_sample_rate)

            # Mel
            if "mel" in artifacts_list:
                img_mel = transformation_mel(audio_chunk).cpu().numpy()[0]
                mel_file_path = os.path.join(output_dir, "mel", label, f"{idx}.npy")
                np.save(mel_file_path, img_mel)

            # CQT
            if "cqt" in artifacts_list:
                img_cqt = transformation_cqt(audio_chunk).cpu().numpy()[0]
                cqt_file_path = os.path.join(output_dir, "cqt", label, f"{idx}.npy")
                np.save(cqt_file_path, img_cqt)

            # Gammatone
            if "gammatone" in artifacts_list:
                img_gamma = transformation_gamma(audio_chunk).cpu().numpy()[0]
                gamma_file_path = os.path.join(output_dir, "gammatone", label, f"{idx}.npy")
                np.save(gamma_file_path, img_gamma)

    meta_file_path = os.path.join(output_dir, f"metadata.csv")
    metadata.to_csv(meta_file_path)
    return

def main():
    parser = create_parser()
    args = parser.parse_args()
    with open(args.config_file) as file:
        args_list = yaml.load(file, Loader=yaml.FullLoader)

    print("Generating dataset as raw data\n")

    out_dir = create_dir(args_list["paths"]["output_dir"])

    sample_rate = args_list["hyperparameters"]["sample_rate"]
    number_of_samples = sample_rate * args_list["hyperparameters"]["number_of_samples"]

    train_metadata_path = args_list["paths"]["train_metadata"]
    test_metadata_path = args_list["paths"]["test_metadata"]
    validation_metadata_path = args_list["paths"]["validation_metadata"]

    interleaved = True if args_list["hyperparameters"]["interleaved"] == 1 else False
    print(f"Generating interleaved metadata..." if interleaved else f"Using original metadata...")

    save_audio = True if args_list["hyperparameters"]["save_audio"] == 1 else False
    print(f"Saving audio files..." if save_audio else f"NOT Saving audio files...")

    # Generate test Dataset
    #print("Generating the test dataset")
    #test_dir = create_dir(os.path.join(out_dir, "test"))
    #generate_dataset_artifacts(test_metadata_path, test_dir, sample_rate, number_of_samples, interleaved=interleaved, save_audio=save_audio)

    # Generate validation Dataset
    #print("Generating the validation dataset")
    #validation_dir = create_dir(os.path.join(out_dir, "validation"))
    #generate_dataset_artifacts(validation_metadata_path, validation_dir, sample_rate, number_of_samples, interleaved=interleaved, save_audio=save_audio)

    # Generate train Dataset
    print("Generating the train dataset")
    train_dir = create_dir(os.path.join(out_dir, "train"))
    generate_dataset_artifacts(train_metadata_path, train_dir, sample_rate, number_of_samples, interleaved=interleaved, save_audio=save_audio)

if __name__ == "__main__":
    main()