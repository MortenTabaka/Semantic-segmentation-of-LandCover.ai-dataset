import tensorflow as tf
import numpy as np
from typing import List, Tuple
import os
from glob import glob
from pandas import DataFrame

from src.data.data_features import ImageFeatures, MaskFeatures
from src.data.image_transformation import ImageTransformator


class Dataset:
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
        self.number_of_classes = number_of_classes

    def generate_datasets(self):
        (
            train_images,
            train_masks,
            val_images,
            val_masks,
            test_images,
            test_masks,
        ) = self.paths_to_images_and_masks()
        train_dataset = self.generate_single_dataset(train_images, train_masks)
        val_dataset = self.generate_single_dataset(val_images, val_masks)
        test_dataset = self.generate_single_dataset(test_images, test_masks)
        return train_dataset, val_dataset, test_dataset

    def generate_single_dataset(
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

    def get_class_balance(self) -> dict:
        each_class_pixel_count = {}

        for class_num in range(self.number_of_classes):
            each_class_pixel_count[class_num] = 0.0

        _, train_masks, _, val_masks, _, test_masks = self.paths_to_images_and_masks()
        all_masks_paths = train_masks + val_masks + test_masks

        for mask_path in all_masks_paths:
            mask = self.mask_features.load_mask_from_drive(mask_path)
            array_of_classes = mask[..., 0]
            one_mask_count = np.unique(array_of_classes, return_counts=True)
            for i in range(len(one_mask_count[0])):
                each_class_pixel_count[one_mask_count[0][i]] += one_mask_count[1][i]

        return each_class_pixel_count

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

    def paths_to_images_and_masks(self) -> Tuple[List[str], ...]:
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

    def get_dataframe_of_previously_calculated_class_balance_class_balance(self) -> DataFrame:
        class_count = self.get_previously_calculated_class_balance()
        class_names = self.get_ordered_class_names()

        df = DataFrame(
            data=class_count.items(),
            index=class_names,
            columns=["class_number", "pixel_count"],
        )
        return df

    @staticmethod
    def get_previously_calculated_class_balance() -> dict:
        return {
            0: 1626435631.0,
            1: 24566909.0,
            2: 925644528.0,
            3: 175769115.0,
            4: 45708873.0,
        }

    @staticmethod
    def get_ordered_class_names() -> List[str]:
        return ["background", "buildings", "woodland", "water", "roads"]
