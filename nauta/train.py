import argparse
import os
import random
import shutil
import torch
import yaml
import numpy as np

from torch import nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, Precision, Recall, F1, ConfusionMatrix

from nauta.tools.utils import create_dir
from nauta.data.dataset import get_dataset
from nauta.model.model import get_model
from nauta.tools.checkpoint import CheckpointManager, Checkpoint
from nauta.tools.train_manager import TrainManager


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


def main():
    """The core of the training execution.
    Initializes all the variables and call the respective methods.
    """
    torch.manual_seed(8)
    random.seed(8)
    np.random.seed(8)

    parser = create_parser()
    args = parser.parse_args()
    with open(args.config_file) as file:
        args_list = yaml.load(file, Loader=yaml.FullLoader)

    print("Start training\n\n")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    num_of_epochs = args_list["hyperparameters"]["epochs"]
    learning_rate = args_list["hyperparameters"]["learning_rate"]

    log_dir = create_dir(os.path.join(args_list["paths"]["output_dir"], "logs"))
    final_model_dir = create_dir(os.path.join(args_list["paths"]["output_dir"], "final_model"))
    checkpoint_dir = create_dir(os.path.join(args_list["paths"]["output_dir"], "checkpoints"))

    # Copy the config file to the output dir
    config_file_name = os.path.basename(args.config_file)
    config_file_path = os.path.join(args_list["paths"]["output_dir"], config_file_name)
    shutil.copyfile(args.config_file, config_file_path)

    # Get the training, validation and test dataloaders.
    train_dataloader, validation_dataloader = get_dataset(args_list)

    # Declare the model.
    model = get_model(model_name="cnn", device=device)
    print("Model Architecture")
    print(summary(model, (3, 64, 63)))

    # Initialise loss funtion + optimizer.
    loss_fn = nn.CrossEntropyLoss()

    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Initialize metrics.
    accuracy = Accuracy(average='macro', num_classes=5)
    precision = Precision(average='macro', num_classes=5)
    recall = Recall(average='macro', num_classes=5)
    f1 = F1(average='macro', num_classes=5)
    confusion_matrix = ConfusionMatrix(num_classes=5)

    metrics = {
        "Accuracy":accuracy,
        "Precision":precision,
        "Recall":recall,
        "F1":f1,
        "ConfusionMatrix":confusion_matrix,
    }

    # Create a checkpoint manager.
    checkpoint_manager = CheckpointManager(
        Checkpoint(model, optimizer), checkpoint_dir, device, max_to_keep=5, keep_best=True
    )
    init_epoch = checkpoint_manager.restore_or_initialize()

    # Create tensorboard writer.
    writer = SummaryWriter(log_dir=log_dir)

    # Add model graph and hyperparams to the logs.
    images, _ = next(iter(train_dataloader))
    writer.add_graph(model, images)

    # Call train routine.
    train_manager = TrainManager(
        model,
        loss_fn,
        optimizer,
        lr_scheduler,
        train_dataloader,
        validation_dataloader,
        num_of_epochs,
        initial_epoch=init_epoch,
        metrics=metrics,
        reference_metric="Accuracy",
        writer=writer,
        device=device
        )

    train_manager.start_train(checkpoint_manager)

    # Close tensorboard writer.
    writer.close()

    # Save the last checkpoint model.
    torch.save(model.state_dict(), os.path.join(final_model_dir, "last.pth"))

    # Save the best accuraccy model.
    checkpoint_manager.load_best_checkpoint()
    torch.save(model.state_dict(), os.path.join(final_model_dir, "best.pth"))


if __name__ == "__main__":
    main()
