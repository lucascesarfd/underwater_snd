import argparse
import os
import torch
import yaml
import numpy as np

from tqdm import tqdm

from nauta.data.dataset import DeeperShip, create_data_loader
from nauta.tools.utils import create_dir
from nauta.data.preprocessing import get_preprocessing_layer

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

def data_saver(dataloader, output_dir, sample_rate, device="cpu"):

    labels_names = list(dataloader.dataset.class_mapping.keys())
    for preprocessing in ["mel", "cqt", "gammatone"]:
        create_dir(os.path.join(output_dir, preprocessing))
        for label in labels_names:
            create_dir(os.path.join(output_dir, preprocessing, label))

    transformation_mel = get_preprocessing_layer("mel", sample_rate)
    transformation_cqt = get_preprocessing_layer("cqt", sample_rate)
    transformation_gamma = get_preprocessing_layer("gammatone", sample_rate)

    for idx, (input_data, target_data) in enumerate(tqdm(dataloader)):
        input_data = input_data.to(device)
        target_data = target_data.to(device)

        #snd = input_data.cpu().numpy()[0]
        label = int(target_data.cpu().numpy()[0])

        # Mel
        img_mel = transformation_mel(input_data).cpu().numpy()[0]
        mel_file_path = os.path.join(output_dir, "mel", labels_names[label], f"{idx}.npy")
        np.save(mel_file_path, img_mel)

        # CQT
        img_cqt = transformation_cqt(input_data).cpu().numpy()[0]
        cqt_file_path = os.path.join(output_dir, "cqt", labels_names[label], f"{idx}.npy")
        np.save(cqt_file_path, img_cqt)

        # Gammatone
        img_gamma = transformation_gamma(input_data).cpu().numpy()[0]
        gamma_file_path = os.path.join(output_dir, "gammatone", labels_names[label], f"{idx}.npy")
        np.save(gamma_file_path, img_gamma)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    with open(args.config_file) as file:
        args_list = yaml.load(file, Loader=yaml.FullLoader)

    print("Generating dataset as raw data\n")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    out_dir = create_dir(args_list["paths"]["output_dir"])

    sample_rate = args_list["hyperparameters"]["sample_rate"]
    number_of_samples = sample_rate * args_list["hyperparameters"]["number_of_samples"]
    batch_size = 1

    pre_processing_type = args_list["preprocessing"]["type"].lower()
    train_metadata_path = args_list["paths"]["train_metadata"]
    test_metadata_path = args_list["paths"]["test_metadata"]
    validation_metadata_path = args_list["paths"]["validation_metadata"]

    #transformation = get_preprocessing_layer(pre_processing_type, sample_rate)

    # Generate test Dataset
    print("Generating the test dataset")
    test_dataset = DeeperShip(test_metadata_path, sample_rate, number_of_samples)
    test_dataloader = create_data_loader(test_dataset, batch_size=batch_size)

    test_dir = create_dir(os.path.join(out_dir, "test"))
    data_saver(test_dataloader, test_dir, sample_rate, device=device)

    # Generate train Dataset
    print("Generating the train dataset")
    train_dataset = DeeperShip(train_metadata_path, sample_rate, number_of_samples)
    train_dataloader = create_data_loader(train_dataset, batch_size=batch_size)

    train_dir = create_dir(os.path.join(out_dir, "train"))
    data_saver(train_dataloader, train_dir, sample_rate, device=device)

    # Generate validation Dataset
    print("Generating the validation dataset")
    validation_dataset = DeeperShip(validation_metadata_path, sample_rate, number_of_samples)
    validation_dataloader = create_data_loader(validation_dataset, batch_size=batch_size)

    validation_dir = create_dir(os.path.join(out_dir, "validation"))
    data_saver(validation_dataloader, validation_dir, sample_rate, device=device)
