import os
import torchaudio

import glob
import os
from pathlib import Path

import numpy as np

from tqdm import tqdm
from nauta.tools.utils import create_dir
from nauta.preprocessing import get_preprocessing_layer


def generate_dataset_artifacts(root_path, target_sample_rate):
    audio_dir = root_path / "audio"
    ship_types = [os.path.basename(d) for d in glob.glob(str(audio_dir / "*"))]

    for preprocessing in ["mel", "cqt", "gammatone"]:
        print(f"Starting with {preprocessing}...")

        # Create the output directory
        preprocessing_dir = root_path / preprocessing
        create_dir(str(preprocessing_dir))

        # Define the transformation
        transformation = get_preprocessing_layer(preprocessing, target_sample_rate)

        for ship in ship_types:
            print(f"Generating data from {ship}")
            ship_dir = create_dir(str(preprocessing_dir / ship))
            audio_files = [Path(d) for d in glob.glob(str(audio_dir / ship / "*.wav"))]
            for audio in tqdm(audio_files):
                signal, sr = torchaudio.load(audio)

                img = transformation(signal).cpu().numpy()[0]
                file_path = Path(ship_dir) / f"{audio.stem}.npy"
                np.save(file_path, img)
    return


def main():

    print("Generating dataset as raw data\n")

    # Generate test Dataset
    print("Generating the test dataset")
    test_path = Path("/workspaces/underwater/dataset/DeeperShip/4k/preprocessed/test")
    sample_rate = 32000
    generate_dataset_artifacts(test_path, sample_rate)

    # Generate validation Dataset
    print("Generating the validation dataset")
    val_path = Path("/workspaces/underwater/dataset/DeeperShip/4k/preprocessed/validation")
    sample_rate = 32000
    generate_dataset_artifacts(val_path, sample_rate)

    # Generate train Dataset
    print("Generating the train dataset")
    train_path = Path("/workspaces/underwater/dataset/DeeperShip/4k/preprocessed/train")
    sample_rate = 32000
    generate_dataset_artifacts(train_path, sample_rate)

if __name__ == "__main__":
    main()