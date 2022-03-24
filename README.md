Semantic segmentation for LandCover.ai dataset
==============================

The project focuses on reproducing the results of the authors of the dataset and will make further efforts to improve the accuracy of the model.

## Create Anaconda Environment

1. To install Anaconda follow [Anaconda documentation](https://docs.anaconda.com/anaconda/install/index.html)

2.  Open terminal in project directory and run command:  ```conda env create --name CHOSEN_ENV_NAME --file tf2.yml```  or  ```conda create --name CHOSEN_ENV_NAME --file pkgs.txt```

3. Run environment with command: ```conda activate CHOSEN_ENV_NAME```

## Prepare dataset. 

1. Open terminal in chosen directory. 

2. Clone repository: `git clone https://github.com/MortenTabaka/Improving-semantic-segmentation-accuracy-for-roads-class-in-LandCover.ai-dataset.git`

3. Change current directory to repository's one with command: `cd Improving-semantic-segmentation-accuracy-for-roads-class-in-LandCover.ai-dataset/`

4. Download and preapare data by using **`make data`** command in repository dir. Dataset will be downloaded and preprocessed according to [these Python files](https://github.com/MortenTabaka/Improving-semantic-segmentation-accuracy-for-roads-class-in-LandCover.ai-dataset/tree/main/src/data). After preparing the data, it is possible to run notebooks. 

5. Run Jupyter with `jupyter-notebook`.

## Jupyter Notebooks

All notebooks can be found [here](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/tree/main/notebooks/exploratory).

### Data exploration

1. [Check the class convention in the mask](https://github.com/MortenTabaka/Improving-semantic-segmentation-accuracy-for-roads-class-in-LandCover.ai-dataset/blob/main/notebooks/exploratory/1.0-Marcin-verify_mask_convention_for_classes.ipynb).

2. [Preparing images for training](https://github.com/MortenTabaka/Improving-semantic-segmentation-accuracy-for-roads-class-in-LandCover.ai-dataset/blob/main/notebooks/exploratory/2.0-Marcin-prepare_data_for_training.ipynb).

### Model exploration

1. [GAN; Generator: MUNet](https://github.com/MortenTabaka/Improving-semantic-segmentation-accuracy-for-roads-class-in-LandCover.ai-dataset/blob/GAN_with_MUnet_generator/notebooks/exploratory/4.0-Marcin-GAN_model_v2.ipynb). It is possible to run training the model, but it gives completely wrong results (needs further development; not completed).

2. DeepLabv3+ architecture:

| Ver. | Backbone | Weights | Frozen convolution base | Loss function | Data augmentation | Train dataset size | Loss weights | mIoU on test dataset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [5.1](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/main/notebooks/exploratory/5.1-Marcin-DeepLabv3%2B_model.ipynb) | Tensorflow Xception | Imagenet | Yes | Sparse Categorical Crossentropy | No | 7470 | No | 0.587 | 
| [5.2](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/main/notebooks/exploratory/5.2-Marcin-DeepLabv3%2B_model.ipynb) | Tensorflow Xception | Imagenet | Yes | Sparse Categorical Crossentropy | Yes | 14940 | No | 0.423 |
| [5.3](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/main/notebooks/exploratory/5.3-Marcin-DeepLabv3%2B_model.ipynb) | Tensorflow Xception | Imagenet | Yes | Sparse Categorical Crossentropy | No | 7470 | Yes | 0.542 |
| [5.4](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/main/notebooks/exploratory/5.4-Marcin-DeepLabv3%2B_model.ipynb) | Modified Xception | Cityscapes | Yes | Sparse Categorical Crossentropy | No | 7470 | No | 0.549 |
| [5.4](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/main/notebooks/exploratory/5.4-Marcin-DeepLabv3%2B_model.ipynb) | Modified Xception | Cityscapes | Yes | Sparse Categorical Crossentropy | No | 7470 | Yes | 0.562 |
| [5.5](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/main/notebooks/exploratory/5.5-Marcin-DeepLabv3%2B_model.ipynb) | Modified Xception | Cityscapes | Yes | Sparse Categorical Crossentropy | No | 7470 | Yes | 0.567 |
| [5.6](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/main/notebooks/exploratory/5.6-Marcin-DeepLabv3%2B_model.ipynb) | Modified Xception | Cityscapes | Yes | Sparse Categorical Crossentropy | No | 7470 | Yes | 0.536 |
| [5.7](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/main/notebooks/exploratory/5.7-Marcin-DeepLabv3%2B_model.ipynb) | Modified Xception | Cityscapes | No | Sparse Categorical Crossentropy | No | 7470 | Yes | 0.359 |
| [5.8](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/main/notebooks/exploratory/5.8-Marcin-DeepLabv3%2B_model.ipynb) | Modified Xception | Cityscapes | Yes | Soft Dice Loss | No | 7470 | No | 0.559 |
| [5.9](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/main/notebooks/exploratory/5.9-Marcin-DeepLabv3%2B_model.ipynb) | Modified Xception | Pascal VOC | Partially | Soft Dice Loss | No | 7470 | No | 0.607 |
| [**5.10**](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/DeepLabv3%2B/notebooks/exploratory/5.10-Marcin-DeepLabv3%2B_model.ipynb) | Modified Xception | Cityscapes | Partially | Soft Dice Loss | No | 7470 | No | **0.718** |
| [5.11](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/DeepLabv3%2B/notebooks/exploratory/5.11-Marcin-DeepLabv3%2B_model.ipynb) | Modified Xception | Cityscapes | Partially | Soft Dice Loss | Yes | 14940 | No | 0.659 |
| [5.12](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/DeepLabv3%2B/notebooks/exploratory/5.12-Marcin-DeepLabv3%2B_model.ipynb) | Modified Xception | Cityscapes | Partially | Soft Dice Loss | Yes | 7470 | No | 0.652 |

## Currently best IoU score

![alt text](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/DeepLabv3%2B/reports/figures/Currently_best_classes_iou.png)

meanIoU = 0.718

Notebook is available [**here**](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/DeepLabv3%2B/notebooks/exploratory/5.10-Marcin-DeepLabv3%2B_model.ipynb).

## References
<a id="1">[1]</a> 
Boguszewski, Adrian and Batorski, Dominik and Ziemba-Jankowska, Natalia and Dziedzic, Tomasz and Zambrzycka, Anna (2021). ["LandCover.ai: Dataset for Automatic Mapping of Buildings, Woodlands, Water and Roads from Aerial Imagery"](https://arxiv.org/abs/2005.02264v2)

<a id="2">[2]</a> 
A. Abdollahi, B. Pradhan, G. Sharma, K. N. A. Maulud and A. Alamri, ["Improving Road Semantic Segmentation Using Generative Adversarial Network,"](https://ieeexplore.ieee.org/document/9416669) in IEEE Access, vol. 9, pp. 64381-64392, 2021, doi: 10.1109/ACCESS.2021.3075951.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
