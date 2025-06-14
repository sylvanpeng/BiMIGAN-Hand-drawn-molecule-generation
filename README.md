# Overview

This package provides CycleGAN and generator implementations used in the`BiMIGAN` [][uvcgan_paper].

`BiMIGAN` introduces an improved method to perform an unpaired image-to-image
style transfer based on a CycleGAN framework. Combined with a new hybrid
generator architecture and a self-supervised pre-training.

This README file provides brief instructions about how to set up the `BiMIGAN`package and reproduce the results.


# Installation & Requirements

## Requirements

`BiMIGAN` was trained on an NVIDIA RTX 4090 GPU with CUDA 11.1, python3.7  under Ubuntu 22.04 LTS. You can create a similar training environment with `conda`

```
conda env create -f contrib/conda_env.yml
```

## Installation

To install the `BiMIGAN` ，run the following command from the source.
```
python setup.py develop --user
```
## Environment Variables Configuration

The `BiMIGAN` framework relies on two key environment variables to manage paths:

`BiMIGAN_DATA`: Specifies the directory containing input datasets (default: `./data`).

`BiMIGAN_OUTDIR`: Defines the output directory for results (default: `./outdir`).

Users must set these variables before execution. 

Example:

```
export UVCGAN_DATA="/path/to/your/data"  
export UVCGAN_OUTDIR="/path/to/your/output"  
```

`BiMIGAN` will look for datasets in the`"${BiMIGAN_DATA}"` directory and will save results under the `"${BiMIGAN_OUTDIR}"`directory. 


# UVCGAN Reproduction

To reproduce tour experiments, follow these steps:

1. Download datasets.
2. Pre-train generators in a BERT-like setup.
3. Train CycleGAN models.
4. Generate translated images and evaluate KID/FID scores.


## 1. Download Datasets

`HNUCM-HDM` and printed molecule dataset can be found and download at:
https://drive.google.com/drive/folders/1MmRwlCdyXwLTZqamDkz0t3eKN6sXCObu?usp=drive_link

download command:
```bash
./scripts/download_dataset.sh
```
The dataset for cyclegan training should be organized in the following structure:
```
data/
└── bimigan_dataset/
    ├── trainA/    # Training images from domain A
    ├── trainB/    # Training images from domain B
    ├── testA/     # Test images from domain A
    └── testB/     # Test images from domain B
```

If you want to pre-train generators on the `ImageNet` dataset,'''``` more details about the origins of these datasets can be found [here](doc/datasets.md).


## 2. Pre-training Generators

To pre-train CycleGAN generators in a BERT-like setup one can use the following similar scripts:

```
scripts/train/bert_imagenet/bert_imagenet-256.py
scripts/train/molgen/bert_printed-256.py
scripts/train/molgen/bert_molgen-256.py
```

For detailed implementation, please refer to the provided scripts. You may either select an appropriate script or modify existing ones to train on your target dataset. The pre-trained generators will be saved under the "${BiMIGAN_OUTDIR}"directory.

For example，to pre-train generators on the `HNUCM-HDM` dataset，execute the following command:

```
python scripts/train/molgen/bert_printed-256.py
```

## 3. Training CycleGAN Generators

Similarly to the generator pre-training, you can use the following command to train the CycleGAN models:
```
python scripts/train/molgen/cyclegan_molgen_256.py
```

More details can be found in the script. The trained CycleGAN models will be saved under the "${BiMIGAN_OUTDIR}" directory.


## 4. Evaluation of the trained model

To perform the style transfer with the trained models `uvcgan` provides a script `scripts/translate_images.py`. Its invocation is simple

```
python scripts/translate_images.py PATH_TO_TRAINED_MODEL -n 100
```
where `-n` parameter controls the number of images from the test dataset to
translate. The original and translated images will be saved under
`PATH_TO_TRAINED_MODEL/evals/final/translated`

The image translation quality was assessed using the [torch_fidelity](https://github.com/toshas/torch-fidelity) package to compute KID and FID metrics. For complete evaluation protocols and implementation details, please refer to the accompanying [benchmarking][benchmarking_repo] repository for the KID/FID evaluation details.

# LICENSE

`BiMIGAN` is distributed under `BSD-2` license.

# Citation
If you use this code for your research, please cite our paper.

