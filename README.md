# pathology_classification
Whole-slide image classification based on preextracted features.

This repository provides tools to tessalate whole-slide images, extract features using pretrained models and to train a classifier based on these preextracted features. Its functionality is divided into three steps:
1. Tessalate: take a whole-slide image and split it into a square, non-overlapping patches.
2. Extract: summarize these patches into a feature vector using a pretrained feature extractor.
3. Hyperparameter sweep: explore configurations of hyperparameters to find optimal settings for the final classifier.
4. Evaluate: train and evaluate a model.

The user needs to supply a dataset of whole-slide images, a manifest csv-file containing the path to the slides and corresponding labels, and a configuration file containing the paths to the manifest file and desired output directory.

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
Create a manifest csv-file containing at least the columns "slide_id" and "slide_path". The slide_id column should contain a unique identifier for every whole-slide image. The slide_path column should be the absolute path to the corresponding slide. In addition, you can supply as many columns with data about these slides as desired. 
```
slide_id, slide_path, binary_label_1, binary_label_2, characteristic_1
example_1, /path/to/example_1.ndpi, 1, 0, lymph_node
...
```

Create a yaml-file in the config/general directory. Fill in the "manifest_file_path" field (the absolute path to the manifest you have created in the previous step) and the "output_dir" field (the path to the directory where the output of the pipeline should be stored). 
```
cd config/general
touch my_settings.yaml
nano my_settings.yaml
```

## Step 1: Tessalate

## Step 2: Extract features

## Step 3: Hyperparameter sweep

## Step 4: Evaluate
