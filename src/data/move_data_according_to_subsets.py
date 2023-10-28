#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 20:03:31 2022

@author: marcin
"""
import glob
import os
import os.path
import shutil

import pandas as pd


def move_data(raw_data_path, processed_data_path):
    """Move data to train/val/test directories."""

    dataSplit = ["train", "val", "test"]

    for whichSplit in dataSplit:
        tilesFolder = raw_data_path + "/tiles"
        moveToSubset = processed_data_path + "/" + whichSplit

        if not os.path.exists(moveToSubset):
            os.makedirs(moveToSubset + "/images/img")
            os.makedirs(moveToSubset + "/masks/img")

        file_with_listed_images = raw_data_path + "/" + whichSplit + ".txt"
        # with open(file_with_listed_images) as f:
        #     imgNames = f.readlines()
        imgNames = pd.read_csv(
            raw_data_path + "/" + whichSplit + ".txt",
            names=["file_name"],
            lineterminator="\n",
        )

        for imgName in imgNames["file_name"]:
            image = imgName + ".jpg"
            org_mask_name = imgName + "_m.png"
            mask = imgName + ".png"

            if os.path.isfile(tilesFolder + "/" + image) and not os.path.isfile(
                moveToSubset + "/images/img/" + image
            ):
                shutil.move(
                    tilesFolder + "/" + image, moveToSubset + "/images/img/" + image
                )

            if os.path.isfile(tilesFolder + "/" + org_mask_name) and not os.path.isfile(
                moveToSubset + "/masks/img/" + mask
            ):
                shutil.move(
                    tilesFolder + "/" + org_mask_name,
                    moveToSubset + "/masks/img/" + mask,
                )


abs_path = os.path.abspath("")

moveFrom = abs_path + "/data/raw"
moveTo = abs_path + "/data/processed"

move_data(moveFrom, moveTo)
