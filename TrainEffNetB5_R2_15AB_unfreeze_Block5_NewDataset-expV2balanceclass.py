import PIL
from keras import models
from keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import os
from tensorflow.keras import callbacks
from keras.callbacks import Callback
import pandas as pd
from keras.utils import generic_utils
import tensorflow as tf


os.environ["CUDA_VISIBLE_DEVICES"]="1"

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 16
epochs = 200

#Train
dataframe = pd.read_csv('/media/tohn/HDD/VISION_dataset/Traindf_fold4_8_v1_Balance_classes.csv')
#validation
valframe = pd.read_csv('/media/tohn/HDD/VISION_dataset/Valdf_fold3_v1.csv')
#from efficientnet.keras import EfficientNetB5 as Net
from tensorflow.keras.applications import EfficientNetB5 as Net
from efficientnet.keras import center_crop_and_resize, preprocess_input
height=width = 456
input_shape = (height, width, 3)
# loading pretrained conv base model
conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape)
# create new model with a new classification layer
x = conv_base.output  
global_average_layer = layers.GlobalAveragePooling2D(name = 'head_pooling')(x)
# Adding BatchNormalization after pooling
batch_norm_layer = layers.BatchNormalization(name='head_batchnorm')(global_average_layer)
# Dropout layer after BatchNormalization
dropout_layer_1 = layers.Dropout(0.50,name = 'head_dropout')(batch_norm_layer)
# Final classification layer
prediction_layer = layers.Dense(15, activation='softmax',name = 'prediction_layer')(dropout_layer_1)
# Combine layers to define the model
model = models.Model(inputs= conv_base.input, outputs=prediction_layer) 
print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))  
#Unfreez block5a_se_excite -to- FC layer
model.trainable = True
set_trainable = False
for layer in model.layers:
    if layer.name == 'block5a_se_excite':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
print('This is the number of trainable layers '
          'after freezing the conv base:', len(model.trainable_weights))  
model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      brightness_range=[0.5,1.5],
      shear_range=0.4,
      zoom_range=0.2,
      horizontal_flip=False,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe = dataframe,
        directory = None,
        x_col = 'Path Crop',
        y_col = 'Sub_class_New',
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='categorical')
test_generator = test_datagen.flow_from_dataframe(
        dataframe = valframe,
        directory = None,
        x_col = 'Path Crop',
        y_col = 'Sub_class_New',
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='categorical')

## Create path to save model
root_model = '/media/tohn/HDD/Model_unlearn/Models_USAI/expV2/R2_balanceclass/models'
os.makedirs(root_model, exist_ok=True)
## Create path to save tensorboard
root_logdir = '/media/tohn/HDD/Model_unlearn/Models_USAI/expV2/R2_balanceclass/Mylogs_tensor'
os.makedirs(root_logdir, exist_ok=True)
def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir, run_id)
### Run TensorBoard 
run_logdir = get_run_logdir(root_logdir)
tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)
   
root_Metrics = '/media/tohn/HDD/Model_unlearn/Models_USAI/expV2/R2_balanceclass/on_epoch_end'
os.makedirs(root_Metrics, exist_ok=True)
class Metrics(Callback):
            def on_epoch_end(self, epochs, logs={}):
                if epochs%20 == 0 and epochs != 0:
                    self.model.save(f'{root_Metrics}/modelEffNetB5_base_Block5a_se_excite_newdatabalance-R2_epoch{epochs}.h5')
                else:
                    self.model.save(f'{root_Metrics}/modelEffNetB5_base_Block5a_se_excite_newdatabalance-R2_last.h5')
                return

# For tracking Quadratic Weighted Kappa score and saving best weights
metrics = Metrics()

## Compile model
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

# Fit model with class weights
model.fit(train_generator, epochs=epochs, 
              validation_data=test_generator,
              callbacks=[metrics, tensorboard_cb])


model.save(f'{root_model}/modelEffNetB5_base_Block5a_se_excite_newdatabalance-R2.h5')
print(f"=============== [INFO]: Save Model Completed >>> {root_model}/modelEffNetB5_base_Block5a_se_excite_newdatabalance-R2.h5 ===============")

