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

# Load the pre-trained model
path_model = '/media/tohn/SSD/Machine-Unlearning-USAI-CCA-2024/models/modelEffNetB5_Unlearning_unfreezeB4_R2.h5'

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense#, Conv2D, BatchNormalization
from tensorflow.keras.applications import EfficientNetB5

# Step 1: โหลดโมเดลเดิมที่มี EfficientNet-B5
old_model = load_model(path_model)

# Step 2: ดึง EfficientNet-B5 จากโมเดลเดิม
efficientnet_b5_layer = old_model.get_layer('efficientnet-b5')  # ดึงเฉพาะ EfficientNet-B5
efficientnet_b5_layer.trainable =  False #True  # สามารถเลือกให้ฝึกต่อได้ หรือจะ freeze ก็ได้

# Step 3: สร้าง input layer ใหม่ที่มีขนาด (224, 224, 3)
new_input = Input(shape=(224, 224, 3), name='new_input')
height=width = 224

# Step 4: ต่อเลเยอร์ EfficientNet-B5 เข้ากับ input ใหม่
x = efficientnet_b5_layer(new_input)
# Step 5: เพิ่มเลเยอร์ Global Average Pooling, Dropout, และเลเยอร์ output (15 classes)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # สามารถปรับ dropout ตามความเหมาะสม
output = Dense(15, activation='softmax', name='prediction_layer')(x)  # 15 classes ใช้ softmax สำหรับ multi-class classification

# Step 6: สร้างโมเดลใหม่
new_model = Model(inputs=new_input, outputs=output)

# Step 7: Compile โมเดลใหม่ (ปรับ optimizer และ loss function ตามที่ต้องการ)
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 8: ดูสรุปโมเดลใหม่
new_model.summary()

#showing before&after freezing
print('This is the number of trainable layers '
      'before freezing the conv base:', len(new_model.trainable_weights))
#conv_base.trainable = False  # freeze เพื่อรักษา convolutional base's weight
for layer in new_model.layers:
    layer.trainable = False
print('This is the number of trainable layers '
      'after freezing the conv base:', len(new_model.trainable_weights))

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
new_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = new_model.fit_generator(
      avoid_error(train_generator),
      steps_per_epoch= len(dataframe)//batch_size,
      epochs=epochs,
      validation_data=avoid_error(test_generator), 
      validation_steps= len(valframe) //batch_size)

### Save history
import pickle
with open('/home/yupaporn/codes/Machine-Unlearning-USAI-CCA-2024/history/Fine-Tune-Machine-Unlearning-V1-R1-200', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
    
from matplotlib import pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()
plt.savefig('/home/yupaporn/codes/Machine-Unlearning-USAI-CCA-2024/history/Fine-Tune-Machine-Unlearning-V1-R1-200.png', bbox_inches='tight')

# 🪐 Save autoencoder model ---------------------------------------------------(✨)

# serialize model to JSON 
model_json = new_model.to_json()
with open("/home/yupaporn/codes/Machine-Unlearning-USAI-CCA-2024/models/Fine-Tune-Machine-Unlearning-V1-R1-200.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
new_model.save_weights("/home/yupaporn/codes/Machine-Unlearning-USAI-CCA-2024/models/Fine-Tune-Machine-Unlearning-V1-R1-200.weights.h5")
print("Saved model to disk")

    

        
        
