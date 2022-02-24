import argparse
import os
import torch
import torchaudio
import torchvision
import random
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchmetrics import Accuracy, Precision, Recall, F1, ConfusionMatrix

from utils import plot_confusion_matrix, create_dir
from dataset import DeepShipDataset, create_data_loader
from model import FeedForwardNet, get_model, pre_processing_layers
from checkpoint import CheckpointManager, Checkpoint


def create_parser():
    # Create the parser
    parser = argparse.ArgumentParser(description="Execute the training routine.")
    true_var = True

    # Add the arguments
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=30,
        help="The number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=25,
        help="The number of elements in a batch",
    )
    parser.add_argument(
        "--sample_rate",
        "-s",
        type=int,
        default=32000,
        help="",
    )
    parser.add_argument(
        "--number_of_samples",
        "-n",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--learning_rate",
        "-l",
        type=float,
        default=0.001,
        help="",
    )
    parser.add_argument(
        "--metadata_file",
        "-m",
        type=str,
        default="/workspaces/underwater/deepship/metadata_10s.csv",
        help="",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="/workspaces/underwater/classifiers/results",
        help="",
    )
    parser.add_argument(
        "--pre_processing_type",
        "-p",
        type=str,
        choices=['mel','gammatone','cqt'],
        default="mel",
        help="",
    )

    return parser


def train_single_epoch(model, train_dataloader, loss_fn, optimizer, device, writer, epoch):
    model.train()
    step = epoch * len(train_dataloader)
    for input_data, target_data, measures_data in tqdm(train_dataloader):

        input_data = input_data.to(device)
        target_data = target_data.to(device)

        # calculate loss
        prediction = model(input_data, measures_data)
        loss = loss_fn(prediction, target_data)

        step += 1
        writer.add_scalar('Loss/train', loss, step)

        # backpropagate error and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Train Loss: {loss.item()}")
    return loss.item()


def validate_single_epoch(model, validation_dataloader, device, writer, epoch, metrics={}):
    model.eval()
    for input_data, target_data, measures_data in tqdm(validation_dataloader):

        input_data = input_data.to(device)
        target_data = target_data.to(device)

        prediction = model(input_data, measures_data)
        for metric in metrics:
            metrics[metric](prediction, target_data)

    for metric in metrics:
        value = metrics[metric].compute()
        if metric == "ConfusionMatrix":
            cm_fig = plot_confusion_matrix(value.numpy(), class_names=validation_dataloader.dataset.class_mapping.keys())
            writer.add_figure(f'Metrics/{metric}', cm_fig, epoch)
        else:
            print(f"Validation {metric}: {value}")
            writer.add_scalar(f'Metrics/{metric}', value, epoch)
        metrics[metric].reset()


def train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, writer, epochs, checkpoint_manager, metrics={}, initial_epoch=0, device='cpu'):

    for epoch in range(initial_epoch, epochs):
        print(f"Epoch {epoch+1}")
        loss = train_single_epoch(model, train_dataloader, loss_fn, optimizer, device, writer, epoch)
        validate_single_epoch(model, validation_dataloader, device, writer, epoch, metrics=metrics)

        # Save a checkpoint.
        checkpoint_manager.save(epoch)

        print("---------------------------")
    print("Finished training")


def main():
    torch.manual_seed(8)
    random.seed(8)
    np.random.seed(8)

    parser = create_parser()
    args = parser.parse_args()

    print("Start training\n\n")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    metadata_file = args.metadata_file
    sample_rate = args.sample_rate
    number_of_samples = sample_rate * args.number_of_samples
    num_of_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    log_dir = create_dir(os.path.join(args.output_dir, "logs"))
    final_model_dir = create_dir(os.path.join(args.output_dir, "final_model"))
    checkpoint_dir = create_dir(os.path.join(args.output_dir, "checkpoints"))
    pre_processing_type = args.pre_processing_type.lower()

    transformation = pre_processing_layers[pre_processing_type](sample_rate)

    # Get the training, validation and test dataloaders.
    file_name = metadata_file.split(".")[0]

    train_dataset = DeepShipDataset(f"{file_name}_train.csv", sample_rate, number_of_samples, transform=transformation)
    train_dataloader = create_data_loader(train_dataset, batch_size=batch_size)

    validation_dataset = DeepShipDataset(f"{file_name}_validation.csv", sample_rate, number_of_samples, transform=transformation)
    validation_dataloader = create_data_loader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Declare the model.
    model = get_model(model_name="physaudio", device=device)
    print("Model Architecture")
    #summary(model, [(1, 64, 63), (1, 5)])
    print()

    # Initialise loss funtion + optimizer.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize metrics.
    accuracy = Accuracy(average='macro', num_classes=6)
    precision = Precision(average='macro', num_classes=6)
    recall = Recall(average='macro', num_classes=6)
    f1 = F1(average='macro', num_classes=6)
    confusion_matrix = ConfusionMatrix(num_classes=6)

    metrics = {
        "Accuracy":accuracy,
        "Precision":precision,
        "Recall":recall,
        "F1":f1,
        "ConfusionMatrix":confusion_matrix,
    }

    # Create a checkpoint manager.
    checkpoint_manager = CheckpointManager(Checkpoint(model, optimizer), checkpoint_dir, device, max_to_keep=5)
    init_epoch = checkpoint_manager.restore_or_initialize()

    # Create tensorboard writer.
    writer = SummaryWriter(log_dir=log_dir)

    # Add model graph and hyperparams to the logs.
    images, _, measures = next(iter(train_dataloader))
    writer.add_graph(model, [images, measures])

    # Call train routine.
    train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, writer, num_of_epochs, checkpoint_manager, metrics=metrics, initial_epoch=init_epoch, device=device)

    # Close tensorboard writer.
    writer.close()

    # Save model.
    torch.save(model.state_dict(), os.path.join(final_model_dir, "final_model.pth"))


if __name__ == "__main__":
    main()
