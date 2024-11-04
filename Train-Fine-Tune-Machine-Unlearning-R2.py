import PIL
# from keras import models
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
import pandas as pd
from efficientnet.keras import EfficientNetB5 as Net
from efficientnet.keras import center_crop_and_resize, preprocess_input
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
from keras import models
from tensorflow.keras.models import Model, model_from_json
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense#, Conv2D, BatchNormalization
from tensorflow.keras.applications import EfficientNetB5

os.environ["CUDA_VISIBLE_DEVICES"]="1"

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


batch_size = 8
epochs = 200

#Train
dataframe = pd.read_csv('/media/tohn/HDD/VISION_dataset/Traindf_fold4_8_v1.csv')
base_dir = '/media/tohn/SSD/Images/Image1'
os.chdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
#validation
valframe = pd.read_csv( '/media/tohn/HDD/VISION_dataset/Valdf_fold3_v1.csv')
validation_dir = os.path.join(base_dir, 'validation')

# # 🍉 load json and create model --------------------

json_file = open('/home/yupaporn/codes/Machine-Unlearning-USAI-CCA-2024/models/Fine-Tune-Machine-Unlearning-V1-R1-200.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# # 🍉 load weights into new model  -------------------

model.load_weights("/home/yupaporn/codes/Machine-Unlearning-USAI-CCA-2024/models/Fine-Tune-Machine-Unlearning-V1-R1-200.weights.h5")
print("Loaded model from disk")

model.summary()

print('This is the number of trainable layers '
      'before freezing the conv base:', len(model.layers[1].trainable_weights))
layers = model.layers[1].layers
for innerlayer in layers:
    if innerlayer.name.startswith("block4"):
#             print(innerlayer.name)
        innerlayer.trainable = True
## Unfreeze FC layer
fc_layer = model.get_layer("prediction_layer")
fc_layer.trainable = True

print('This is the number of trainable layers '
      'after freezing the block5a_se_excite Layer:', len(model.layers[1].trainable_weights))

height = width = 224

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
        directory = train_dir,
        x_col = 'Path Crop',
        y_col = 'Sub_class_New',
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='categorical')
test_generator = test_datagen.flow_from_dataframe(
        dataframe = valframe,
        directory = validation_dir,
        x_col = 'Path Crop',
        y_col = 'Sub_class_New',
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='categorical')

def avoid_error(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass
        
#Training
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      avoid_error(train_generator),
      steps_per_epoch= len(dataframe)//batch_size,
      epochs=epochs,
      validation_data=avoid_error(test_generator), 
      validation_steps= len(valframe) //batch_size)

### Save history
import pickle
with open('/home/yupaporn/codes/Machine-Unlearning-USAI-CCA-2024/history/Fine-Tune-Machine-Unlearning-V1-R2-500', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
    
from matplotlib import pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()
plt.savefig('/home/yupaporn/codes/Machine-Unlearning-USAI-CCA-2024/history/Fine-Tune-Machine-Unlearning-V1-R2-500.png', bbox_inches='tight')

# 🪐 Save autoencoder model ---------------------------------------------------(✨)

# serialize model to JSON 
model_json = model.to_json()
with open("/home/yupaporn/codes/Machine-Unlearning-USAI-CCA-2024/models/Fine-Tune-Machine-Unlearning-V1-R2-500.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/yupaporn/codes/Machine-Unlearning-USAI-CCA-2024/models/Fine-Tune-Machine-Unlearning-V1-R2-500.weights.h5")
print("Saved model to disk")

    

        
        
