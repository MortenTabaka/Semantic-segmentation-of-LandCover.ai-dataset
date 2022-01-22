#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 17:38:33 2021

@author: marcin

source:
https://ericbassett.tech/cookiecutter-data-science-crash-course/
"""

import pandas as pd
import shutil

import glob
import os
import os.path
import cv2


class DataProcessor:
    """
    Class for reading, processing, and writing data.
    """
    def __init__(self):
        pass
    
    def split_images(self, raw_data_path):
        """"
        Split each original image and its corresponding mask into 512x512
        tiles and shuffle them.
        
        Source: LandCover.ai
        """
        
        IMGS_DIR = raw_data_path + "/images"
        MASKS_DIR = raw_data_path + "/masks"
        OUTPUT_DIR = raw_data_path + "/tiles"
        
        TARGET_SIZE = 512
        
        img_paths = glob.glob(os.path.join(IMGS_DIR, "*.tif"))
        mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.tif"))
        
        img_paths.sort()
        mask_paths.sort()
        
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
                    img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
                    mask_tile = mask[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
        
                    if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
                        out_img_path = os.path.join(OUTPUT_DIR, 
                                                    "{}_{}.jpg".format(img_filename, k))
                        cv2.imwrite(out_img_path, img_tile)
        
                        out_mask_path = os.path.join(OUTPUT_DIR, 
                                                     "{}_{}_m.png".format(mask_filename, k))
                        cv2.imwrite(out_mask_path, mask_tile)
        
                    k += 1

            print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))
        

    def move_data(raw_data_path, processed_data_path):
        """Move data to train/val/test directories."""
        
        dataSplit = ['train', 'val', 'test']
        
        for whichSplit in dataSplit:
            
            moveFrom = raw_data_path + '/tiles'
            moveTo = processed_data_path + '/' + whichSplit
           
            if not os.path.exists(moveTo):
                
                os.makedirs(moveTo + '/images/img')
                os.makedirs(moveTo + '/masks/img')
            
            imgsNames = pd.read_csv(raw_data_path + '/' + whichSplit + '.txt',
                                    sep='\n', names=['file_name'])
            
            for imgName in imgsNames['file_name']:
                
                image = imgName + '.jpg'
                org_mask_name = imgName + '_m.png'
                mask = imgName + '.png'
                
                if os.path.isfile(moveFrom + '/' + image):
                
                    shutil.move(moveFrom + '/' + image,
                                moveTo + '/images/img/' + image)
                    
                if os.path.isfile(moveFrom + '/' + org_mask_name):
                
                    shutil.move(moveFrom + '/' + org_mask_name,
                                moveTo + '/masks/img/' + mask)
        