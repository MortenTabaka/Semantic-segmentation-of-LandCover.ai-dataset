#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 16:23:30 2021

@author: marcin
"""

import os


def create_dir(path_list):

    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)


paths = [
    "./data",
    "./data/raw",
    "./data/processed",
    "./data/external",
    "./data/interim",
]

create_dir(paths)
