Semantic segmentation of LandCover.ai dataset
==============================

The LandCover dataset consists of aerial images of urban and rural areas of Poland. The project focuses on the application of various neural networks for semantic segmentation, including the reconstruction of the neural network implemented by the authors of the dataset. 

## Create a project environment

To get Tensorflow and required drivers (if not installed), please visit [tutorial](https://www.youtube.com/watch?v=PnK1jO2kXOQ) and its [instruction](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/tensorflow-install-jul-2020.ipynb).

Project was created using:
* Python 3.8
* Tensorflow 2.5

### Possible errors

I. `Could not load dynamic library 'libcudnn.so.8'`

Solution:

Visit [discussion](https://github.com/tensorflow/tensorflow/issues/45200#issue-7514283790) or `conda install -c nvidia cudnn`

## Prepare dataset

1. Open terminal in chosen directory. 

2. Clone repository: `git clone https://github.com/MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset.git`

3. Change current directory to repository's one with command: `cd Semantic-segmentation-of-LandCover.ai-dataset/`

4. Download and preapare data by using **`pip install gdown && make data`** command in repository directory. Dataset will be downloaded and preprocessed according to [these Python files](https://github.com/MortenTabaka/Improving-semantic-segmentation-accuracy-for-roads-class-in-LandCover.ai-dataset/tree/main/src/data). After preparing the data, it is possible to run notebooks. 

5. Run Jupyter `jupyter notebook`

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

## Currently best mIoU score

![alt text](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/DeepLabv3%2B/reports/figures/Currently_best_classes_iou.png)

meanIoU = 0.718

Notebook is available [**here**](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/DeepLabv3%2B/notebooks/exploratory/5.10-Marcin-DeepLabv3%2B_model.ipynb).

Weights generating best mIOU on test set are [**available to download from Gdrive.**](https://drive.google.com/drive/folders/1MyJ0_lQxBW7ekOzuVaBG4BRIpbaA-7xj?usp=sharing)

Run below script in the project folder to create a model and load the weights.

```
#get local path to the project
ABS_PATH = %pwd
slash_idx = [idx for idx,ch in enumerate(ABS_PATH) if ch=='/']
ABS_PATH = ABS_PATH[:slash_idx[-2]]

#add folder with models to sys.path to be able to make imports in Jupyter
module_path = ABS_PATH + '/src/models'
if module_path not in sys.path:
    sys.path.append(module_path)

from deeplabv3plus import Deeplabv3


IMG_SIZE = 512
NUM_CLASSES = 5


def get_deeplab_model(weights=None, activation=None):
    
    """Returns pre-configured Deeplabv3+ used in the project.
    
    Args:
    weights: one of 'pascal_voc' (pre-trained on pascal voc),
            'cityscapes' (pre-trained on cityscape) or None (random initialization)
    activation: optional activation to add to the top of the network.
            One of 'softmax', 'sigmoid' or None
    """
    
    model = Deeplabv3(
        weights=weights,
        classes=NUM_CLASSES,
        backbone='xception',
        OS=16,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        activation=activation)
        
    return model
    

# path to downloaded folder with weights when put in the project's data folder 
PATH_TO_DOWNLOADED_CHECKPOINT = ABS_PATH + '/data/0.718mIOU/checkpoint'

#since we are loading weights
#there is no need to load pre-trained Cityscapes or Pascal weights
model = get_deeplab_model()

#load downloaded weights
model.load_weights(PATH_TO_DOWNLOADED_CHECKPOINT)

```

## TODO

- [x] [Add functionality to download and preprocess dataset with `make data` command.](https://github.com/MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset/tree/main/src/data)
- [x] [Check annotations on masks and how pixel class information is communicated.](https://github.com/MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset/blob/main/notebooks/exploratory/1.0-Marcin-verify_mask_convention_for_classes.ipynb).
- [x] Create a pipeline for the data to be feed into the model.
- [ ] GAN model with MUnet as backbone according to research paper (Jupyter notebook).
- [x] DeepLabv3+ model with modified Xception and pretrained weights (Jupyter notebook).
    * [ ] Modify architecture by increasing split connections and adding more layers to decoder.  
- [ ] Check meticulously if annotations in provided masks are correctly assgined.
   * [x] [Create notebook for finding predicted mask with low IoU, to help determine which ground truth masks should be removed due to low quality.](https://github.com/MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset/blob/DeepLabv3%2B/notebooks/exploratory/6.1-Marcin-check_quality_of_provided_masks.ipynb)
   * [ ] Exclude low quality masks from train and validation sets and retrain model from start (downloaded weights trained on Cityscapes).

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

## Citation
If you use this software, please cite it using these metadata.
```
@software{Tabaka_Semantic_segmentation_of_2021,
author = {Tabaka, Marcin Jarosław},
license = {Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)},
month = {11},
title = {{Semantic segmentation of LandCover.ai dataset}},
url = {https://github.com/MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset},
year = {2021}
}
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
