To use the dockerized version of Tensorflow,
first follow the official Nvidia installation guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

## DockerHub Image for Linux
The ready-to-use image is accessible on Docker Hub.

### Pull the image
```
docker pull mortentabaka/landcover_semantic_segmentation_with_deeplabv3plus:latest
```
### Run the image in interactive mode
```
docker run --gpus all -it -p 8888:8888 mortentabaka/landcover_semantic_segmentation_with_deeplabv3plus:latest

```
### Run the image and create files locally
```
export PROJECT_PATH_LOCALLY="/path/to/local/code/directory" &&
git clone https://github.com/MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset.git "$PROJECT_PATH_LOCALLY" &&
docker run --gpus all -it -p 8888:8888 -v $PROJECT_PATH_LOCALLY:/app/ mortentabaka/landcover_semantic_segmentation_with_deeplabv3plus:latest
```

## Dockerfile - Tensorflow GPU
Clone the repository:
```
git clone git@github.com:MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset.git && cd Semantic-segmentation-of-LandCover.ai-dataset/
```

Build the docker image with the project's [Dockerfile](https://github.com/MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset/blob/main/Dockerfile):
```
docker build -t landcover_semantic_segmentation .
```
An official image was used as a base: https://hub.docker.com/layers/tensorflow/tensorflow/2.5.1-gpu-jupyter/images/sha256-5cdcd4446fc110817e5f6c5784eba6254258632036b426b9f194087e200f8a96?context=explore

Run the Jupyter Notebook with:
```
docker run --gpus all -it --rm -p 8888:8888 -v $(pwd):/app landcover_semantic_segmentation
```

## Dockerfile - Tensorflow CPU (not tested)
Clone the repository
```
git clone git@github.com:MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset.git && cd Semantic-segmentation-of-LandCover.ai-dataset/
```

In [Dockerfile](https://github.com/MortenTabaka/Semantic-segmentation-of-LandCover.ai-dataset/blob/main/Dockerfile)
change tensorflow image name to `tensorflow/tensorflow:2.5.1-jupyter`.

Build a image.
```
docker build -t landcover_semantic_segmentation .
```

To run the image, do not use flag `--gpus all`:

```
docker run -it --rm -p 8888:8888 -v $(pwd):/app landcover_semantic_segmentation
```

### If port is already in use

If port `8888` is already in use, then change its value, e.g. `-p 5555:8888`.
Remember to manually replace port in a link to the chosen value:

Would be: `http://127.0.0.1:8888/?token=...`

Should be: `http://127.0.0.1:5555/?token=...`