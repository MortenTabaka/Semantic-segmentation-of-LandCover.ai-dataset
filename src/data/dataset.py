import tensorflow as tf
import numpy as np
from typing import List
import os
from glob import glob

from src.data.data_features import ImageFeatures, MaskFeatures
from src.data.image_transformation import ImageTransformator


class DataLoader:
    """
    Provides functionality to load processed images and masks.
    """

    def __init__(
        self,
        processed_images_path: str,
        image_height: int,
        image_width: int,
        number_of_classes: int,
        batch_size: int,
    ):
        """
        Class for loading images and masks.
        Args:
            processed_images_path: absolute path to processed images. By default, it is ./data/processed
        """
        self.processed_images_path = processed_images_path[:-1] + processed_images_path[
            -1
        ].replace("/", "")
        self.image_features = ImageFeatures(image_height, image_width)
        self.mask_features = MaskFeatures(image_height, image_width, number_of_classes)
        self.image_transformator = ImageTransformator(image_height, image_width)
        self.batch_size = batch_size

    def generate_dataset(self):
        (
            train_images,
            train_masks,
            val_images,
            val_masks,
            test_images,
            test_masks,
        ) = self.paths_to_images_and_masks()
        train_dataset = self.generator(train_images, train_masks)
        val_dataset = self.generator(val_images, val_masks)
        test_dataset = self.generator(test_images, test_masks)
        return train_dataset, val_dataset, test_dataset

    def generator(
        self,
        images_paths: List[str],
        masks_paths: List[str],
        data_transformation: bool = False,
        augmentation: bool = False,
        augmentation_factor: int = 2,
    ) -> tf.data.Dataset:
        """
        Returns

        Args:
        image_list: list of paths to each image
        mask_list: list of paths to corresponding masks of images (sorted)
        data_transformation: decides whether images and masks will be randomly transformed
        augmentation: decides whether dataset will be increased
        augmentation_factor: Factor by which number of data images will be incremented.
        """
        dataset = tf.data.Dataset.from_tensor_slices((images_paths, masks_paths))

        if data_transformation:
            dataset = dataset.map(
                self.load_single_transformed_image_and_mask,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            dataset = dataset.map(
                self.load_single_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        if augmentation and augmentation_factor > 1:
            for _ in range(augmentation_factor - 1):
                dataset_to_concat = dataset.map(
                    self.load_single_transformed_image_and_mask,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
                dataset = dataset.concatenate(dataset_to_concat)

        return dataset

    def load_single_image_and_mask(self, image_path: str, mask_path: str) -> tf.image:
        image = self.image_features.load_image_from_drive(image_path)
        mask = self.mask_features.load_mask_from_drive(mask_path)
        return image, mask

    def load_single_transformed_image_and_mask(
        self, image_path: str, mask_path: str
    ) -> tf.image:
        image = self.image_features.load_image_from_drive(image_path)
        mask = self.mask_features.load_mask_from_drive(mask_path)
        return self.image_transformator.get_randomly_transformed_image_and_mask(
            image, mask
        )

    def paths_to_images_and_masks(self):
        train_images = sorted(
            glob(os.path.join(self.processed_images_path, "train/images/img/*"))
        )
        train_masks = sorted(
            glob(os.path.join(self.processed_images_path, "train/masks/img/*"))
        )
        val_images = sorted(
            glob(os.path.join(self.processed_images_path, "val/images/img/*"))
        )
        val_masks = sorted(
            glob(os.path.join(self.processed_images_path, "val/masks/img/*"))
        )
        test_images = sorted(
            glob(os.path.join(self.processed_images_path, "test/images/img/*"))
        )
        test_masks = sorted(
            glob(os.path.join(self.processed_images_path, "test/masks/img/*"))
        )

        return train_images, train_masks, val_images, val_masks, test_images, test_masks

    def _get_path_to_subset_of_dataset(self, which_dataset: str) -> str:
        """
        Returns path to
        Args:
            which_dataset: Subset of images created during preprocessing images via Make. By default: train, val, test.

        Returns:
        """
        which_dataset = which_dataset.replace("/", "").lower()
        if self.processed_images_path[-1] == "/":
            return self.processed_images_path + which_dataset
        return self.processed_images_path + "/" + which_dataset
