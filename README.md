# Underwater Vessel Audio Classification Net
The UVACNet is a Deep Learning model destined to classify the underwater audio generated by vessels in the sea.

## Dev Guidelines
There is a `Dockerfile` available to facilitate the use of Docker to development.

Alternativelly, as this repository is entirely based on python, you can make use of `venv`. To install the `requirements`, run the following command:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## How to execute
First it is important to have the DeeperShip dataset files available. They will be made available freely on the [DeeperShip Repository]().

Once you have the files, just create a `config` file containing the hyperparameters and paths to the training execution. An example can be found [here](./config_files/default.yaml).

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
