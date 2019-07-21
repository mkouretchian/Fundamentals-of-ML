#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File:   normalization.py
Desc:   Resizes image to 100x100 pixels and saves images as bmp image  

"""
import cv2 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image 
import os


trainNormal = []
def NormalizeImage(trainData):
    	for x in range(len(trainData)):    
    		img = trainData[x]
    		rows = img.shape[0]
    		cols = img.shape[1]
    		res = cv2.resize(img,(100, 100), interpolation = cv2.INTER_CUBIC)
    		trainNormal.append(res)
    		plt.imsave('Training_Images/filename_' + str(x) + '.png', trainNormal[x], cmap=cm.gray)
    		image = Image.open('Training_Images/filename_' + str(x) + '.png')
    		image.save('Training_Images/filename_' + str(x) + '.bmp')   		
    		os.remove('Training_Images/filename_' + str(x) + '.png')

     
     
  
