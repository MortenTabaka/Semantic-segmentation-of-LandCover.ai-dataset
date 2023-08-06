# Project installation guide

## Open terminal in chosen directory. 

### Clone the repository
```
git clone git@github.com:MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset.git && cd Semantic-segmentation-of-LandCover.ai-dataset/
```
### Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### Create new environment with Python 3.8 and jupyter.
```
conda create --name landcover_semantic_segmentation jupyter python=3.8
```
### Activate the environment.
```
conda activate landcover_semantic_segmentation
```
### Add Jupyter support for the environment.
```
conda install -c conda-forge nb_conda
```
### Install Cudatoolkit, Cudnn and required PIP packages.
```
conda install -c conda-forge cudatoolkit=11.3.1
conda install -c nvidia cudnn
pip install -r requirements.txt
```
### Set up automation for system paths configuration.
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
### Install Tensorflow with only CPU support
```
conda install -c anaconda tensorflow
```
### Install Tensorflow with GPU support
```
conda install -c anaconda tensorflow-gpu
```
#### Check installation 
CPU: 
```
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
["If a tensor is returned, you've installed TensorFlow successfully."](https://www.tensorflow.org/install/pip?hl=en)

GPU:
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
["If a list of GPU devices is returned, you've installed TensorFlow successfully."](https://www.tensorflow.org/install/pip?hl=en)

Success of Tensorflow2's installation is highly depend on your OS and Nvidia drivers, therefore above guide will not always be correct.

I will be experimenting with customizing [Tensorflow's Docker image](https://www.tensorflow.org/install/docker?hl=en), to create a docker image designed for this project.

### Register created Anaconda environment for Jupyter.
```
python -m ipykernel install --user --name tensorflow --display-name "Python 3.8 (tensorflow)"
```
Now you should be able to change kernel in Jupyter Notebook.
#### Create dataset and install pip packages with one command
```
make dataset
``` 
*During the process, pip packages specified in requirements.txt, will be installed.*

### [Notebook for checking installation](https://github.com/MortenTabaka/main/notebooks/testing/check_tensorflow_installation.ipynb)
Filepath: `/notebooks/testing/check_tensorflow_installation.ipynb`

# My environemnt 
## OS
* Debian 12

## Anaconda environment
* Python 3.8
* Tensorflow 2.5 (with GPU)
* Cudnn 8.9.2.26 (` conda list cudnn`)
* Cudatoolkit 11.3.1 (`conda list cudatoolkit`)

## Nvidia
* GPU driver version: 515.65.01 
* CUDA Version: 11.7  

# Sources
* https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/tensorflow-install-jul-2020.ipynb
* https://www.tensorflow.org/install/pip?hl=en
