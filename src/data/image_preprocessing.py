#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 17:38:33 2021

@author: marcin

source:
https://ericbassett.tech/cookiecutter-data-science-crash-course/
"""

import glob
import os
import os.path
from pathlib import Path
from typing import Union

import cv2


class ImagePreprocessor:
    """
    Class for reading, processing, and writing data.
    """

    def __init__(self, path_to_folder_with_input_images: Union[Path, str]):
        self.path_to_folder_with_input_images = path_to_folder_with_input_images

    def split_dataset_images(self):
        """Split each original image and its corresponding mask into 512x512
        tiles and shuffle them.

        Source: LandCover.ai
        """

        IMGS_DIR = self.path_to_folder_with_input_images + "/images"
        MASKS_DIR = self.path_to_folder_with_input_images + "/masks"
        OUTPUT_DIR = self.path_to_folder_with_input_images + "/tiles"

        TARGET_SIZE = 512

        img_paths = glob.glob(os.path.join(IMGS_DIR, "*.tif"))
        mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.tif"))

        img_paths.sort()
        mask_paths.sort()

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
            img_filename = os.path.splitext(os.path.basename(img_path))[0]
            mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]

            k = 0
            for y in range(0, img.shape[0], TARGET_SIZE):
                for x in range(0, img.shape[1], TARGET_SIZE):
                    img_tile = img[y : y + TARGET_SIZE, x : x + TARGET_SIZE]
                    mask_tile = mask[y : y + TARGET_SIZE, x : x + TARGET_SIZE]

                    if (
                        img_tile.shape[0] == TARGET_SIZE
                        and img_tile.shape[1] == TARGET_SIZE
                    ):
                        out_img_path = os.path.join(
                            OUTPUT_DIR, "{}_{}.jpg".format(img_filename, k)
                        )

                        if not os.path.isfile(out_img_path):
                            cv2.imwrite(out_img_path, img_tile)

                        out_mask_path = os.path.join(
                            OUTPUT_DIR, "{}_{}_m.png".format(mask_filename, k)
                        )

                        if not os.path.isfile(out_mask_path):
                            cv2.imwrite(out_mask_path, mask_tile)

                    k += 1

            print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))

    def split_custom_images_before_prediction(
        self,
        target_size: int,
        path_to_output_folder: Union[Path, str],
    ):

        img_paths = glob.glob(
            os.path.join(self.path_to_folder_with_input_images, "*.tif")
        )
        img_paths += glob.glob(
            os.path.join(self.path_to_folder_with_input_images, "*.jpg")
        )
        img_paths += glob.glob(
            os.path.join(self.path_to_folder_with_input_images, "*.png")
        )
        img_paths.sort()

        if not os.path.exists(path_to_output_folder):
            os.makedirs(path_to_output_folder)

        for i, img_path in enumerate(img_paths):
            img_filename = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path)

            k = 0
            for y in range(0, img.shape[0], target_size):
                for x in range(0, img.shape[1], target_size):
                    img_tile = img[y : y + target_size, x : x + target_size]

                    if (
                        img_tile.shape[0] == target_size
                        and img_tile.shape[1] == target_size
                    ):
                        out_img_path = os.path.join(
                            path_to_output_folder, "{}_{}.jpg".format(img_filename, k)
                        )

                        if not os.path.isfile(out_img_path):
                            cv2.imwrite(out_img_path, img_tile)
                    k += 1

            print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))
