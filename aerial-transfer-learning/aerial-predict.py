# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 00:49:21 2023

@author: Ayobami
"""


import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

from mrcnn.visualize import display_instances, display_top_masks
from mrcnn.utils import extract_bboxes

from mrcnn.utils import Dataset
from matplotlib import pyplot as plt

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import mrcnn.config
import mrcnn.model


from mrcnn import model as modellib, utils
from PIL import Image, ImageDraw
CLASS_NAMES = ['BG', 'building', 'vegetation', 'flood', 'car', 'road']

class AConfig(mrcnn.config.Config):
	# define the name of the configuration
    

    NAME = "aerial_cfg_coco"

    NUM_CLASSES = 6
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
	
	
  
	
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=AConfig(),
                             model_dir=os.getcwd())
model.load_weights(filepath="aerial_mask_rcnn_trained.h5", 
                   by_name=True)
image = cv2.imread("aerial/images/test/000000000570.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = model.detect([image], verbose=0)
results = results[0]
display_instances(image, results['rois'], results['masks'], 
                  results['class_ids'], CLASS_NAMES, results['scores'])



