import argparse
import glob
import os
from pathlib import Path

import torchaudio
import numpy as np

from tqdm import tqdm
from nauta.tools.utils import create_dir
from nauta.dataset.preprocessing import get_preprocessing_layer

def create_parser():
    """Create the parser object.

    Returns:
        parser: The generated parser object with arguments
    """
    parser = argparse.ArgumentParser(description="Preprocess the VTUAD dataset.")

    parser.add_argument(
        "-d",
        "--root_dir",
        type=str,
        help="The root dir of the desired scenario from VTUAD dataset.",
        required=True
    )

    return parser


def generate_dataset_artifacts(root_path):
    audio_dir = root_path / "audio"
    ship_types = [os.path.basename(d) for d in glob.glob(str(audio_dir / "*"))]

    for preprocessing in ["mel", "cqt", "gammatone"]:
        print(f"Starting with {preprocessing}...")

        # Create the output directory
        preprocessing_dir = root_path / preprocessing
        create_dir(str(preprocessing_dir))

        # Define the transformation. VTUAD uses 32000 as sample rate.
        transformation = get_preprocessing_layer(preprocessing, 32000)

        for ship in ship_types:
            print(f"Generating data from {ship}")
            ship_dir = create_dir(str(preprocessing_dir / ship))
            audio_files = [Path(d) for d in glob.glob(str(audio_dir / ship / "*.wav"))]
            for audio in tqdm(audio_files):
                signal, sr = torchaudio.load(str(audio))

                img = transformation(signal).cpu().numpy()[0]
                file_path = Path(ship_dir) / f"{audio.stem}.npy"
                np.save(file_path, img)
    return


def main():

    print("Generating preprocessed files from VTUAD dataset\n")

    parser = create_parser()
    args = parser.parse_args()
    root_dir = Path(args.root_dir)

    # Generate test Dataset
    print("Generating the test dataset")
    test_path = root_dir / "test"
    generate_dataset_artifacts(test_path)

    # Generate validation Dataset
    print("Generating the validation dataset")
    val_path = root_dir / "validation"
    generate_dataset_artifacts(val_path)

    # Generate train Dataset
    print("Generating the train dataset")
    train_path = root_dir / "train"
    generate_dataset_artifacts(train_path)

if __name__ == "__main__":
    main()