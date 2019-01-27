#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 20:25:19 2019

@author: matthewpotts
"""
import os
import shutil
import pandas as pd

train_path = '/Users/matthewpotts/Pills-with-Siamese-Networks/data/faces/training'

image_path = '/Users/matthewpotts/Downloads/dc'

df = pd.read_csv("/Users/matthewpotts/Downloads/groundTruthTable.csv")

folder_names = df.ref_images.unique()

for i in range(len(folder_names)):
    folder_nm = 'p' + str(i)
    path = os.path.join(train_path, folder_nm)
    os.mkdir(path)
    
    sub = df.loc[df['ref_images'] == folder_names[i]]
    files = sub['cons_images'].values
    for i in range(len(files)):
        source = os.path.join(image_path, files[i])
        shutil.copy(source, path)







