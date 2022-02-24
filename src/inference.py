import torch
import torchaudio

from dataset import DeepShipDataset, create_data_loader
from model import FeedForwardNet, get_model


class_mapping = ['tug', 'tanker', 'cargo', 'passengership']


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
        is_correct = (predicted_index == target)
    return predicted, expected, is_correct


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    metadata_file = "/workspaces/Underwater/DeepShip/metadata_10s.csv"
    sample_rate = 32000
    number_of_samples = sample_rate * 1
    num_of_epochs = 10
    batch_size = 25

    # load back the model
    model = get_model(model_name="cnn", device=device)
    state_dict = torch.load("final_model.pth")
    model.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    deep_ship = DeepShipDataset(metadata_file, mel_spectrogram, sample_rate, number_of_samples)

    max_data = 1000
    predictions = 0
    total = max_data
    for input, target in deep_ship:
        input.unsqueeze_(0)
        # make an inference
        predicted, expected, is_correct = predict(model, input, target,
                                      class_mapping)
        predictions += is_correct

        if max_data == 1:
            break
        else:
            max_data -= 1 

    print(f"Train Accuracy: {predictions/total}")