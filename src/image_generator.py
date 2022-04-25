import argparse
import os
import torch
import yaml
import numpy as np

from tqdm import tqdm

from dataset import DeeperShip, create_data_loader
from utils import create_dir
from model import pre_processing_layers

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

def data_saver(dataloader, output_dir, device="cpu"):

    labels_names = list(dataloader.dataset.class_mapping.keys())
    for label in labels_names:
        create_dir(os.path.join(output_dir, label))

    for idx, (input_data, target_data) in enumerate(tqdm(dataloader)):
        input_data = input_data.to(device)
        target_data = target_data.to(device)

        img = input_data.cpu().numpy()[0]
        label = int(target_data.cpu().numpy()[0])

        file_path = os.path.join(output_dir, labels_names[label], f"{idx}.npy")

        np.save(file_path, img)



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

    transformation = pre_processing_layers[pre_processing_type](sample_rate)

    # Generate test Dataset
    print("Generating the test dataset")
    test_dataset = DeeperShip(
        test_metadata_path, sample_rate, number_of_samples, transform=transformation
    )
    test_dataloader = create_data_loader(test_dataset, batch_size=batch_size)

    test_dir = create_dir(os.path.join(out_dir, "test"))
    data_saver(test_dataloader, test_dir, device=device)

    # Generate train Dataset
    print("Generating the train dataset")
    train_dataset = DeeperShip(
        train_metadata_path, sample_rate, number_of_samples, transform=transformation
    )
    train_dataloader = create_data_loader(train_dataset, batch_size=batch_size)

    train_dir = create_dir(os.path.join(out_dir, "train"))
    data_saver(train_dataloader, train_dir, device=device)

    # Generate validation Dataset
    print("Generating the validation dataset")
    validation_dataset = DeeperShip(
        validation_metadata_path, sample_rate, number_of_samples, transform=transformation
    )
    validation_dataloader = create_data_loader(validation_dataset, batch_size=batch_size)

    validation_dir = create_dir(os.path.join(out_dir, "validation"))
    data_saver(validation_dataloader, validation_dir, device=device)
