# Nauta Project
This repository contains a pipeline for classifying underwater acoustic data using Deep Learning methods. This code currently supports ResNet, a VGG-based net, and a Feed-Forward network to classify the data.

## Environment Setup
There is a `Dockerfile` available to facilitate the use of Docker to development.

Alternativelly, as this repository is entirely based on python, you can make use of `venv`. To install the `requirements.txt`, run the following command:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Also, this project was made as a python package, so after the dependencies install, run the following command to install the `nauta` package:

```bash
pip install .
```

## Dataset Preparation
First it is important to have the [VTUAD Dataset](https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data) files available. For more information of this dataset generation, see the [ONC Dataset Generation Pipeline](https://github.com/lucascesarfd/onc_dataset) repository.

Once you have the data, there are two ways of running this code:

1. Running using the WAV files and processing on each iteration.

2. Running from a preprocessed version of the data.

For the first option, just the dataset download will be sufficient. For the second option, the user must generate the preprocessed version of the dataset first.

### Preprocessing the data
**If you want to run directly from `WAV` files, ignore this section**
To preprocess the data, run the [preprocessing_generator.py](./nauta/tools/preprocessing_generator.py) file. The only needed argument is the path to the VTUAD data from the needed scenario:

```bash
python preprocessing_generator.py -d <DATASET_ROOT_DIR>/inclusion_2000_exclusion_4000/
```

This will produce the folders `mel`, `cqt`, and `gammatone` under the scenario folder.

## How to execute

Create a `config` file containing the hyperparameters and paths to the training execution. An example of running from `WAV` files can be found on [vtuad.yaml](./config_files/vtuad.yaml). An example of running from preprocessed files can be found on [vtuadfeature.yaml](./config_files/vtuadfeature.yaml)

With the `config` file updated, run the following command:

```bash
python ./nauta/train.py <path to config file>
```

The training session can be followed using Tensorboard. The logs are saved on the `output_dir` indicated on the `config` file.

After the execution, an inference (using the test dataset) can be performed to evaluate the results. To perform an inference, just run the following command:

```bash
python ./src/inference.py <path to config file>
```

The best weights saved by the training session will be considered on the execution.

## Reference
The results from this work were published at IEEE Access, at the following reference:

[An Investigation of Preprocessing Filters and Deep Learning Methods for Vessel Type Classification With Underwater Acoustic Data](https://ieeexplore.ieee.org/document/9940921)

```bibtex
@article{domingos2022investigation,
  author={Domingos, Lucas C. F. and Santos, Paulo E. and Skelton, Phillip S. M. and Brinkworth, Russell S. A. and Sammut, Karl},
  journal={IEEE Access}, 
  title={An Investigation of Preprocessing Filters and Deep Learning Methods for Vessel Type Classification With Underwater Acoustic Data}, 
  year={2022},
  volume={10},
  number={},
  pages={117582-117596},
  doi={10.1109/ACCESS.2022.3220265}}
```

A complete literature review containing the background knowledge of this work is available on the following reference:

[A Survey of Underwater Acoustic Data Classification Methods Using Deep Learning for Shoreline Surveillance](https://www.mdpi.com/1424-8220/22/6/2181)

```bibtex
@article{domingos2022survey,
  author={Domingos, Lucas C. F. and Santos, Paulo E. and Skelton, Phillip S. M. and Brinkworth, Russell S. A. and Sammut, Karl},
  title={A Survey of Underwater Acoustic Data Classification Methods Using Deep Learning for Shoreline Surveillance},
  volume={22},
  ISSN={1424-8220},
  url={http://dx.doi.org/10.3390/s22062181},
  DOI={10.3390/s22062181},
  number={6},
  publisher={MDPI AG},
  journal={Sensors},
  year={2022},
  month={Mar},
  pages={2181}
}
```
