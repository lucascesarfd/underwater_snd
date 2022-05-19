from torch.utils.data import DataLoader

from nauta.preprocessing import get_preprocessing_layer

from nauta.one_stage.dataset import DeeperShipFeature, DeeperShip
from nauta.two_stage.dataset import VesselDeeperShipFeature

def create_data_loader(data, batch_size, shuffle=True):
    """Creates a pytorch dataloader from a Dataset.

    Args:
        data (Dataset): The desired dataset.
        batch_size (int): The size of the mini batch.
        shuffle (bool, optional): Indicates if the data needs to be shuffled. Defaults to True.

    Returns:
        DataLoader: The generated dataloader.
    """
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    return loader


def get_dataset(config):
    """Returns the desired dataloaders for validation and train.

    Args:
        model_name (str, required): The name of the dataset according to the documentation.

    Returns:
        DataLoader, DataLoader : The train and the validation dataloaders, respectively.
    """
    if config["dataset"]["type"] == "deepershipfeature":
        batch_size = config["hyperparameters"]["batch_size"]
        train_dataset_path = config["dataset"]["train_root_path"]
        validation_dataset_path = config["dataset"]["validation_root_path"]
        num_of_classes = config["model"]["num_of_classes"]

        # Get the training and validation.
        train_dataset = DeeperShipFeature(train_dataset_path, num_of_classes=num_of_classes)
        train_dataloader = create_data_loader(train_dataset, batch_size=batch_size)

        validation_dataset = DeeperShipFeature(validation_dataset_path, num_of_classes=num_of_classes)
        validation_dataloader = create_data_loader(validation_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, validation_dataloader
    else:
        sample_rate = config["hyperparameters"]["sample_rate"]
        number_of_samples = sample_rate * config["hyperparameters"]["number_of_samples"]
        batch_size = config["hyperparameters"]["batch_size"]

        train_metadata_path = config["dataset"]["train_metadata"]
        validation_metadata_path = config["dataset"]["validation_metadata"]

        pre_processing_type = config["preprocessing"]["type"].lower()
        transformation = get_preprocessing_layer(pre_processing_type, sample_rate)

        # Get the training, validation and test dataloaders.
        train_dataset = DeeperShip(
            train_metadata_path, sample_rate, number_of_samples, transform=transformation
        )
        train_dataloader = create_data_loader(train_dataset, batch_size=batch_size)

        validation_dataset = DeeperShip(
            validation_metadata_path, sample_rate, number_of_samples, transform=transformation
        )
        validation_dataloader = create_data_loader(validation_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, validation_dataloader