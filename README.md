Semantic segmentation of LandCover.ai dataset
==============================

The LandCover dataset consists of aerial images of urban and rural areas of Poland. The project focuses on the application of various neural networks for semantic segmentation, including the reconstruction of the neural network implemented by the authors of the dataset. 

## Sample results

### High mIoU
![meanIoU_100 0_percent__4952](https://user-images.githubusercontent.com/29732555/201164572-ee6b56b6-b87f-4480-a52d-943678f5245b.png)
![meanIoU_73 03_percent__3929](https://user-images.githubusercontent.com/29732555/201164620-34ecbb4c-b6d4-4385-ac3b-0a9540842589.png)
![meanIoU_63 92_percent__5447](https://user-images.githubusercontent.com/29732555/201164668-920369ed-1c7c-45b6-96e0-6142ac71f1ba.png)

### Low mIoU
![meanIoU_6 12_percent__3183](https://user-images.githubusercontent.com/29732555/201164960-118c1efa-fb5b-496e-b8b6-2609c461a92f.png)
![meanIoU_23 39_percent__5703](https://user-images.githubusercontent.com/29732555/201165001-045e3f7a-9dac-4bce-b0f2-e457130f6f3c.png)


# [Installation guide](https://github.com/MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset/blob/main/INSTALLATION.md)

# Jupyter Notebooks

### [Templates](https://github.com/MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset/tree/main/notebooks/templates)

### [Experiments](https://github.com/MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset/tree/main/notebooks/exploratory)

## Data exploration

1. [Check the class convention in the mask](https://github.com/MortenTabaka/Improving-semantic-segmentation-accuracy-for-roads-class-in-LandCover.ai-dataset/blob/main/notebooks/exploratory/1.0-Marcin-verify_mask_convention_for_classes.ipynb).

2. [Preparing images for training](https://github.com/MortenTabaka/Improving-semantic-segmentation-accuracy-for-roads-class-in-LandCover.ai-dataset/blob/main/notebooks/exploratory/2.0-Marcin-prepare_data_for_training.ipynb).

## Model exploration

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
| [**5.10**](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/main/notebooks/exploratory/5.10-Marcin-DeepLabv3%2B_model.ipynb) | Modified Xception | Cityscapes | Partially | Soft Dice Loss | No | 7470 | No | **0.718** |
| [5.11](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/DeepLabv3%2B/notebooks/exploratory/5.11-Marcin-DeepLabv3%2B_model.ipynb) | Modified Xception | Cityscapes | Partially | Soft Dice Loss | Yes | 14940 | No | 0.659 |
| [5.12](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/DeepLabv3%2B/notebooks/exploratory/5.12-Marcin-DeepLabv3%2B_model.ipynb) | Modified Xception | Cityscapes | Partially | Soft Dice Loss | Yes | 7470 | No | 0.652 |

## Currently best mIoU score

![alt text](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/main/reports/figures/Currently_best_classes_iou.png)

meanIoU = 0.718

Notebook is available [**here**](https://github.com/MortenTabaka/Semantic-segmentation-for-LandCover.ai-dataset/blob/main/notebooks/exploratory/5.10-Marcin-DeepLabv3%2B_model.ipynb).



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
- [x] Migrate notebooks' functionality to separate Python modules.
- [ ] Resolve issue with environment's installation: https://github.com/MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset/issues/1 

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
    |   |   ├── architectures      <- Model architectures available for training
    │   │   ├── predict_model.py   
    │   │   └── model_builder.py
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
