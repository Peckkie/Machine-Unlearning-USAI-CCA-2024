import PIL
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFile
from efficientnet.keras import center_crop_and_resize, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


ImageFile.LOAD_TRUNCATED_IMAGES = True

def Data_generator(height, width, BATCH_SIZE, dataframe, valframe):
    
    train_datagen = ImageDataGenerator(
          rescale=1./255,
          rotation_range=30,
          width_shift_range=0.2,
          height_shift_range=0.2,
          brightness_range=[0.5, 1.5],
          shear_range=0.4,
          zoom_range=0.2,
          horizontal_flip=False,
          fill_mode='nearest')

    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe = dataframe,
        directory = None,
        x_col = 'Path Crop',
        y_col = 'Sub_class_New',
        target_size = (height, width),
        batch_size=BATCH_SIZE,
        color_mode= 'rgb',
        class_mode='categorical')

    val_generator = valid_datagen.flow_from_dataframe(
        dataframe = valframe,
        directory = None,
        x_col = 'Path Crop',
        y_col = 'Sub_class_New',
        target_size = (height, width),
        batch_size=BATCH_SIZE,
        color_mode= 'rgb',
        class_mode='categorical')
    
    return train_generator, val_generator 

