from torch import nn
from nauta.one_stage.model import FeedForwardNet, CNNNetwork, CNNNetworkCQT, ResNet18

models_list = {
    "feedforward": FeedForwardNet,
    "cnn": CNNNetwork,
    "cnncqt":CNNNetworkCQT,
    "resnet18":ResNet18,
}


def set_parameter_requires_grad(model, feature_extracting=False):
    """Sets parameters to not require gradients if feature_extracting.

    Args:
        model (models): torchvision model
        feature_extracting (bool): True to feature extraction, False to finetunning
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def init_weights(model):
    """Initializes the weights of a model.

    Args:
        model (nn.Model): The model to be initialized.
    """
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            nn.init.zeros_(model.bias)
    if isinstance(model, nn.Conv2d):
        nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            nn.init.zeros_(model.bias)


def get_model(config, device="cpu"):
    """Returns the desired model initialized.

    Args:
        model_name (str, optional): The name of the model according to the documentation. Defaults to "FeedForward".
        device (str, optional): The device to load the tensors. Defaults to "cpu".

    Returns:
        nn.Model: The loaded model object.
    """
    model_name = config["model"]["name"]
    model_depth = config["model"]["cnn_depth"]
    input_channels = config["model"]["input_channels"]
    num_of_classes = config["model"]["num_of_classes"]

    model = models_list[model_name.lower()](
        model_depth=model_depth,
        input_channels=input_channels,
        out_classes=num_of_classes
    ).to(device)

    model.apply(init_weights)

    return model
