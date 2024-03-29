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

from nauta.dataset import get_dataset
from nauta.model import get_model
from nauta.tools import CheckpointManager, Checkpoint, TrainManager, create_dir


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

    num_of_classes = args_list["model"]["num_of_classes"]
    input_channels = args_list["model"]["input_channels"]

    optim_type = args_list["optim"]["type"]
    early_stop = False if args_list["optim"]["early_stop"] == 0 else True

    num_of_epochs = args_list["hyperparameters"]["epochs"]
    learning_rate = args_list["hyperparameters"]["learning_rate"]
    lr_schd_gamma = args_list["hyperparameters"]["lr_schd_gamma"]

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
    model = get_model(args_list, device=device)
    print("Model Architecture")
    print(summary(model, (input_channels, 95, 126)))

    # Initialise loss funtion + optimizer.
    loss_fn = nn.CrossEntropyLoss()

    if optim_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optim_type == "nadam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_schd_gamma)

    # Initialize metrics.
    accuracy = Accuracy(average='macro', num_classes=num_of_classes)
    accuracy_micro = Accuracy(average='micro', num_classes=num_of_classes)
    accuracy_weight = Accuracy(average='weighted', num_classes=num_of_classes)
    precision = Precision(average='macro', num_classes=num_of_classes)
    recall = Recall(average='macro', num_classes=num_of_classes)
    f1 = F1(average='macro', num_classes=num_of_classes)
    confusion_matrix = ConfusionMatrix(num_classes=num_of_classes)

    metrics = {
        "Accuracy":accuracy,
        "AccuracyMicro":accuracy_micro,
        "AccuracyWeighted":accuracy_weight,
        "Precision":precision,
        "Recall":recall,
        "F1":f1,
        "ConfusionMatrix":confusion_matrix,
    }

    # Create a checkpoint manager.
    checkpoint_manager = CheckpointManager(
        Checkpoint(model, optimizer), checkpoint_dir, device, max_to_keep=5, keep_best=True
    )
    last_epoch = checkpoint_manager.restore_or_initialize()
    init_epoch = last_epoch + 1 if last_epoch != 0 else 0

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
        device=device,
        early_stop=early_stop,
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
