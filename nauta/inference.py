import argparse
import numpy as np
import os
import random
import torch
import yaml

from torchmetrics import Accuracy, Precision, Recall, F1, ConfusionMatrix, PrecisionRecallCurve
from tqdm import tqdm

from nauta.data.dataset import DeeperShipDataset, create_data_loader
from nauta.model.model import get_model, pre_processing_layers
from nauta.tools.utils import plot_confusion_matrix, create_dir, plot_pr_curve


def create_parser():
    """Create the parser object.

    Returns:
        parser: The generated parser object with arguments
    """
    parser = argparse.ArgumentParser(description="Execute the inference routine.")

    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        default="/workspaces/underwater/dev/underwater_snd/config_files/default.yaml",
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
    data_info.append(f"Dataset Size: {len(dataloader.dataset)}")
    data_info.append(f"Metrics:")

    for input_data, target_data in tqdm(dataloader):
        input_data = input_data.to(device)
        target_data = target_data.to(device)

        prediction = model(input_data)
        for metric in metrics:
            metrics[metric](prediction, target_data)

    for metric in metrics:
        value = metrics[metric].compute()
        if metric == "ConfusionMatrix":
            cm_fig = plot_confusion_matrix(
                value.numpy(), class_names=dataloader.dataset.class_mapping.keys()
            )
            cm_fig.savefig(os.path.join(eval_dir, "confusion.svg"))
        if metric == "PrecisionRecallCurve":
            cm_fig = plot_pr_curve(
                value.numpy(), class_names=dataloader.dataset.class_mapping.keys()
            )
            cm_fig.savefig(os.path.join(eval_dir, "pr_curve.svg"))
        else:
            print(f"Test {metric}: {value}")
            data_info.append(f"  {metric}: {value}")
        metrics[metric].reset()

        # save info into txt file.
    with open(os.path.join(eval_dir, "metrics.txt"), 'w') as f:
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

    print(f"Start inference using {device}\n")

    sample_rate = args_list["hyperparameters"]["sample_rate"]
    number_of_samples = sample_rate * args_list["hyperparameters"]["number_of_samples"]
    batch_size = args_list["hyperparameters"]["batch_size"]
    pre_processing_type = args_list["preprocessing"]["type"].lower()
    test_metadata_path = args_list["paths"]["test_metadata"]
    eval_dir = create_dir(os.path.join(args_list["paths"]["output_dir"], "evaluation"))

    # Initialize the dataset
    transformation = pre_processing_layers[pre_processing_type](sample_rate)
    test_dataset = DeeperShipDataset(
        test_metadata_path, sample_rate, number_of_samples, transform=transformation
    )
    dataloader = create_data_loader(test_dataset, batch_size=batch_size)

    # Initialize the model
    model = get_model(model_name="cnn", device=device)
    model_weights = os.path.join(args_list["paths"]["output_dir"], "final_model", "best.pth")
    state_dict = torch.load(model_weights)
    model.load_state_dict(state_dict)

    # Initialize the metrics.
    accuracy = Accuracy(average='macro', num_classes=5)
    precision = Precision(average='macro', num_classes=5)
    recall = Recall(average='macro', num_classes=5)
    f1 = F1(average='macro', num_classes=5)
    confusion_matrix = ConfusionMatrix(num_classes=5)
    pr_curve = PrecisionRecallCurve(num_classes=5)

    metrics = {
        "Accuracy":accuracy,
        "Precision":precision,
        "Recall":recall,
        "F1":f1,
        "ConfusionMatrix":confusion_matrix,
        "PrecisionRecallCurve":pr_curve,
    }

    evaluate(model, dataloader, metrics, eval_dir, device=device)
