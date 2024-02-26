# pathology_classification
Binary whole-slide image classification based on preextracted features.

This repository provides tools to tessalate whole-slide images, extract features using pretrained models and to train a classifier based on these preextracted features. Its functionality is divided into three steps:
1. Tessalate: take a whole-slide image and split it into a square, non-overlapping patches.
2. Extract: summarize these patches into a feature vector using a pretrained feature extractor.
3. Hyperparameter sweep: explore configurations of hyperparameters to find optimal settings for the final classifier.
4. Evaluate: train and evaluate a final classifier to make binary classifications based on the preextracted features.

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

Step 4: Create a fresh conda environment with Python and R.
```
conda create -n pathology_classification python r-base
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

Create a yaml-file in the config directory. Fill in the "manifest_file_path" field (the absolute path to the manifest you have created in the previous step) and the "output_dir" field (the path to the directory where the output of the pipeline should be stored). 
```
cd config/general
touch my_settings.yaml
nano my_settings.yaml
```

## Step 1: Tessellate
In this step, whole-slide images are preprocessed into square, non-overlapping patches of a fixed size. These images are saved to disk for faster processing during step 2. 

The most important parameters during this step are the extraction level and patch size. The extraction level is the level of magnification at which the patches are extracted, following the OpenSlide API. This means that level 0 is the highest level of magnification (e.g. 40x), level 1 is the next lowest (e.g. 20x) etc. The patch size is the size of the image that is extracted at the specified level of magnification. 

The /source/tessellate/tessellate.py script performs the preprocessing for all required magnification levels and patch sizes in one run. Different extractor models require different patch sizes (e.g. the ResNet50 extractor requires patches of 256x256). The magnification level can be set arbitrarily, except in the case when a feature extractor model (used in step 2) is pretrained on a specific magnification level (e.g., HIPT is pretrained on 20x). In the "extractor_models" section of the config file, the used extractor models are specified along with their required patch sizes and levels of magnification. 

To run this step, use the following command:
```
python source/tessellate/tessellate.py --config config/my_settings.yaml
```

In the supplied output directory, this will create the following directories and subdirectories: 
- tiles
  - extraction_level=1_patch_dimensions=[256,256]
    - case_1
      - 0_0.jpg
      - 0_1024.jpg
    - case_2
      ...
- patch_coordinates
  - extraction_level=1_patch_dimensions=[256,256]
    - case_1.json
    - case_2.json

The patch_coordinates directory contains a .json-file for every case, with the relative location, origin and size of every extracted patch. These are used later for generating heatmaps. 

## Step 2: Extract features

## Step 3: Hyperparameter sweep

## Step 4: Evaluate
