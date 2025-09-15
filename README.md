# sentence_simplification

This repository contains scripts to augment and evaluate datasets for sentence simplification tasks. Various augmentation methods are applied to the model outputs of different models, and the augmented datasets are then evaluated using metrics like SARI, LENS, LENS-SALSA, and BERTScore.

## Folder Structure Src

augment.py: Script to augment the dataset by adding a various augmentation method to the `model_output` of jsonl files of various models.
    Input: Folder path containing subfolders of various models having jsonl files.
evaluate.py: Script to evaluate the augmented dataset using various metrics.
    Input: Folder path containing augmented csv files.
rank_correlation.py: Script to calculate the rank correlation, rmse between top 3 models on original and augmented datasets. It also calculates if the winner is preserved after augmentation.
    Input: Folder path containing evaluated csv files and file names.

## How to run

1. Clone the repository
2. Install the required packages using pip install -r requirements.txt
3. Run the augment.py script to augment the dataset
4. Run the evaluate.py script to evaluate the augmented dataset

## How to choose machine on Google Colab

[Colab GPUs: Features and Pricing Guide](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/#s2-pricing-approach)

| GPU Model | Architecture | Launch Date | VRAM    | Website |
|-----------|--------------|-------------|---------|---------|
| V100      | Volta        | 6/21/17     | 16 GB   | [Details](https://www.nvidia.com/en-us/data-center/v100/) |
| T4        | Turing       | 9/13/18     | 15 GB   | [Details](https://www.nvidia.com/en-us/data-center/tesla-t4/) |
| A100      | Ampere       | 5/14/20     | 40 GB   | [Details](https://www.nvidia.com/en-us/data-center/a100/) |
| L4        | Ada Lovelace | 3/21/23     | 22.5 GB | [Details](https://www.nvidia.com/en-us/data-center/l4/) |
| H100      | Hopper       | 3/22/23     | 80 GB   | [Details](https://www.nvidia.com/en-us/data-center/h100/) |
| B100      | Blackwell    | 2024        | 80 GB   | [Details](https://www.nvidia.com/en-us/data-center/b100/) |