# pathology_classification
Whole-slide image classification based on preextracted features.

This repository provides tools to tessalate whole-slide images, extract features using pretrained models and to train a classifier based on these preextracted features. Its functionality is divided into three steps:
1. Tessalate: take a whole-slide image and split it into a square, non-overlapping patches.
2. Extract: extract features using a pretrained feature extractor.
3. Hyperparameter sweep: explore configurations of hyperparameters to find optimal settings for the final classifier.
4. Evaluate: train and evaluate a model on the supplied dataset.

## Repository structure
![Repository structure](repository_structure.png)

## Installation
Step 1: Install OpenSlide using the instructions in the [documentation](https://openslide.org/api/python/#installing). 

Step 2: Follow the instructions on the [Weights & Biases website](https://docs.wandb.ai/quickstart) to sign up for an account and setup the API token on your system. 

Step 3: Clone this repository:
```
git clone https://github.com/RensterMaat/pathology_classification.git
cd pathology_classification
```

Step 4 (Optional): Create a fresh conda environment:
```
conda create -n pathology_classification python
```

Step 5: Install this repository using pip:
```
pip install . 
```

## What you need to supply

## Step 1: Tessalate

## Step 2: Extract features

## Step 3: Hyperparameter sweep

## Step 4: Evaluate
