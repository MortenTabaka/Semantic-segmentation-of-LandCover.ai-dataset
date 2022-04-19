#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 20:03:31 2022

@author: marcin
"""
import pandas as pd
import shutil

import glob
import os
import os.path
import cv2


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

move_data('/home/marcin/git_workspace/roads_semantic_segmentation_in_LandCover.ai_dataset/data/raw', '/home/marcin/git_workspace/roads_semantic_segmentation_in_LandCover.ai_dataset/data/processed')