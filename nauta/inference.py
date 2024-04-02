import argparse
import numpy as np
import os
import random
import torch
import yaml

from torchmetrics import Accuracy, Precision, Recall, F1, ConfusionMatrix
from torchsummary import summary
from tqdm import tqdm

from nauta.dataset import get_split_dataloader
from nauta.model import get_model
from nauta.tools.utils import plot_confusion_matrix, create_dir


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


def evaluate(model, dataloader, metrics, eval_dir, device='cpu'):
    """Perform an evaluation on the loaded model

    Args:
        model (nn.Module): The model to be used for evaluation.
        dataloader (Dataset): The dataloader object for the test dataset.
        metrics (Dict): A dict containing the name and object of the metrics from torchmetrics.
        eval_dir (os.Path): The path where the artifacts will be saved
        device (str, optional): The device to load the tensors. Defaults to 'cpu'.
    """
    model.eval()
    data_info = []
    data_info.append(f"dataset_size,{len(dataloader.dataset)}")

    for input_data, target_data in tqdm(dataloader):
        input_data = input_data.to(device)
        target_data = target_data.to(device)

        prediction = model(input_data)
        for metric in metrics:
            metrics[metric].update(prediction, target_data)

    for metric in metrics:
        value = metrics[metric].compute()
        if metric == "ConfusionMatrix":
            cm_fig_norm = plot_confusion_matrix(
                value.cpu().detach().numpy(), class_names=dataloader.dataset.class_mapping.keys()
            )
            cm_fig_norm.savefig(os.path.join(eval_dir, "confusion.svg"))
            cm_fig = plot_confusion_matrix(
                value.cpu().detach().numpy(), class_names=dataloader.dataset.class_mapping.keys(), normalize=False
            )
            cm_fig.savefig(os.path.join(eval_dir, "confusion_not_norm.svg"))
        else:
            print(f"Test {metric}: {value}")
            data_info.append(f"{metric.lower()},{value}")
        metrics[metric].reset()

    # save info into txt file.
    with open(os.path.join(eval_dir, "metrics.csv"), 'w') as f:
        for line in data_info:
            f.write(f"{line}\n")


if __name__ == "__main__":
    torch.manual_seed(8)
    random.seed(8)
    np.random.seed(8)

    parser = create_parser()
    args = parser.parse_args()
    with open(args.config_file) as file:
        args_list = yaml.load(file, Loader=yaml.FullLoader)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    num_of_classes = args_list["model"]["num_of_classes"]
    input_channels = args_list["model"]["input_channels"]
    eval_dir = create_dir(os.path.join(args_list["paths"]["output_dir"], "evaluation"))

    print(f"\n\nStarting inference\n")
    print(f"Saving at {eval_dir}...")

    # Initialize the dataset
    dataloader = get_split_dataloader(args_list, split="test")

    # Declare and initialize the model
    model = get_model(args_list, device=device)
    model_weights = os.path.join(args_list["paths"]["output_dir"], "final_model", "best.pth")
    state_dict = torch.load(model_weights)
    model.load_state_dict(state_dict)

    # Initialize the metrics.
    accuracy = Accuracy(average='macro', num_classes=num_of_classes).to(device)
    precision = Precision(average='macro', num_classes=num_of_classes).to(device)
    recall = Recall(average='macro', num_classes=num_of_classes).to(device)
    f1 = F1(average='macro', num_classes=num_of_classes).to(device)
    confusion_matrix = ConfusionMatrix(num_classes=num_of_classes).to(device)

    metrics = {
        "Accuracy":accuracy,
        "Precision":precision,
        "Recall":recall,
        "F1":f1,
        "ConfusionMatrix":confusion_matrix,
    }

    evaluate(model, dataloader, metrics, eval_dir, device=device)
