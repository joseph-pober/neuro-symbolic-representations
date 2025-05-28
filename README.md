# TEST

# Project Overview

This repository contains various Python scripts related to autoencoder experiments, data generation, and visualization.

## File Structure

| File | Description |
|------|------------|
| `aes-old.py` | Older, deprecated version of `aes.py`. |
| `aes.py` | Holds all versions of traditional autoencoders used. |
| `comparer.py` | Deprecated attempt at learning properties through comparison-based training. |
| `data.py` | Helper functions for data processing, label generation, and dataset handling. |
| `data_main.py` | Manages the generation of data when executed. |
| `directories.py` | Stores paths to important external files such as models and training histories. |
| `exp_propertyAE.py` | Property autoencoder with various configurations for experiments. |
| `exp_splitNN.py` | Deprecated prototype for `exp_propertyAE.py`. |
| `exp_supervised_xy.py` | Proto property autoencoder for testing supervised learning of feature vectors. |
| `generator.py` | High-level data generator for experiment datasets. |
| `helper.py` | Deprecated helper functions. |
| `main.py` | Primary script for setting up and running experiments. |
| `nn.py` | Deprecated, generic neural network model. |
| `shapes.py` | Low-level data generator for image shape drawing. |
| `split_nn.py` | Variant of `exp_splitNN.py`, with convolutional layers for image preprocessing. |
| `test.py` | Miscellaneous script for testing different parameter configurations. |
| `vis.py` | Helper functions for visualizing training histories, model structures, and neural activations. |
| `vis_main.py` | Manages visualization by calling functions from `vis.py`. |

## Usage

Run `main.py` to set up and execute experiments. Use `vis_main.py` for visualizations. Data generation is managed by `data_main.py` and `generator.py`.

Feel free to explore the scripts and modify them for your experiments!
