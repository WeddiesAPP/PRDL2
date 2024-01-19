# Deep Learning Models for MEG data

This repository contains the implementation of deep learning models for time MEG Data. The models are designed to work with both 1D and 2D time series data and have been tested on the "Final Project data" dataset, including cross and intra-patient data.

## Overview

The project focuses on developing Convolutional Long Short-Term Memory (CLSTM) and Recurrent Neural Network (RNN) models for MEG data classification. It includes preprocessing scripts for both 1D and 2D data, training/testing scripts, and a grid search script for hyperparameter tuning.

## Prerequisites

Ensure you have the following prerequisites installed:

- Python 3.x
- PyTorch
- h5py
- tqdm
- matplotlib
- scikit-learn

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
