import argparse
import os
import torch
import torchaudio
import torchvision

from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from dataset import DeepShipDataset, create_data_loader
from model import FeedForwardNet, get_model
from utils import get_accuracy, get_precision, get_recall, get_f1
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
        default="/workspaces/Underwater/DeepShip/metadata_10s.csv",
        help="",
    )
    parser.add_argument(
        "--generate_dataset",
        "-g",
        action="store_const",
        default=not(true_var),
        const=true_var,
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="/workspaces/Underwater/classifiers/results",
        help="",
    )

    return parser


def train_single_epoch(model, data_loader, loss_fn, optimizer, device, writer, epoch):
    model.train()
    step = epoch * len(data_loader)
    for input_data, target_data in tqdm(data_loader):

        input_data = input_data.to(device)
        target_data = target_data.to(device)

        # calculate loss
        prediction = model(input_data)
        loss = loss_fn(prediction, target_data)

        step += 1
        writer.add_scalar('Loss/train', loss, step)

        # backpropagate error and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Train Loss: {loss.item()}")
    return loss.item()


def validate_single_epoch(model, validation_dataloader, device, writer, epoch):
    correct = 0
    total = 0
    model.eval()
    for input_data, target_data in tqdm(validation_dataloader):

        input_data = input_data.to(device)
        target_data = target_data.to(device)

        prediction = model(input_data)
        total += target_data.size(0)

        correct += (prediction.argmax(1) == target_data).float().sum()

    accuracy = correct / total
    writer.add_scalar('Accuracy/validation', accuracy, epoch)

    print(f"Validation Accuracy: {accuracy}")


def train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, writer, epochs, checkpoint_manager, initial_epoch=0, device='cpu'):

    for epoch in range(initial_epoch, epochs):
        print(f"Epoch {epoch+1}")
        loss = train_single_epoch(model, train_dataloader, loss_fn, optimizer, device, writer, epoch)
        validate_single_epoch(model, validation_dataloader, device, writer, epoch)

        # Save a checkpoint.
        checkpoint_manager.save(epoch)

        print("---------------------------")
    print("Finished training")


def main():
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
    log_dir = os.path.join(args.output_dir, "logs")
    final_model_dir = os.path.join(args.output_dir, "final_model")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    deep_ship = DeepShipDataset(metadata_file, mel_spectrogram, sample_rate, number_of_samples)

    # Get the training dataset.
    train_dataloader, validation_dataloader = create_data_loader(deep_ship, batch_size, validation_split=0.3)

    # Declare the model.
    model = get_model(model_name="cnn", device=device)
    print("Model Architecture")
    summary(model, (1, 64, 63))
    print()

    # Initialise loss funtion + optimizer.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create a checkpoint manager.
    checkpoint_manager = CheckpointManager(Checkpoint(model, optimizer), checkpoint_dir, device, max_to_keep=5)
    init_epoch = checkpoint_manager.restore_or_initialize()

    # Create tensorboard writer.
    writer = SummaryWriter(log_dir=log_dir)

    # Add model graph and hyperparams to the logs.
    images, _ = next(iter(train_dataloader))
    writer.add_graph(model, images)
    h_params = {'learning rate': learning_rate, 'batch size': batch_size, "number of epochs": num_of_epochs}
    metrics = {'Accuracy/validation': None, 'Loss/train': None}

    # Call train routine.
    train(model, train_dataloader, validation_dataloader, loss_fn, optimizer, writer, num_of_epochs, checkpoint_manager, initial_epoch=init_epoch, device=device)

    # Add hparams and close tensorboard writer.
    writer.add_hparams(h_params, metrics)
    writer.close()

    # Save model.
    torch.save(model.state_dict(), os.path.join(final_model_dir, "final_model.pth"))


if __name__ == "__main__":
    main()
