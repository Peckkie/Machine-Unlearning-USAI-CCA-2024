import tensorflow as tf
from skimage.io import imread
from keras.utils import generic_utils
from keras import layers
from keras import models
from tensorflow.keras import optimizers
from keras.optimizers import Adam
import efficientnet.tfkeras as efn
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.layers import Concatenate, Input, Conv2D, BatchNormalization
#load Check point
from tensorflow.keras.models import load_model




def build_modelB5_unlearn(fine_tune):
    """
    :param fine_tune (bool): Whether to train the hidden layers or not.
    """
    
    conv_base = efn.EfficientNetB5(weights='imagenet')
    # Define the input shapes for the two images
    img_height, img_width, channels = 456, 456, 3
    input_shape = (img_height, img_width, channels)
    # Define the input layers for the two images
    input_img1 = Input(shape=(img_height, img_width, channels))
    input_img2 = Input(shape=(img_height, img_width, channels))
    # Concatenate the two input images along the channel axis
    concatenated_input = Concatenate(axis=-1)([input_img1, input_img2])

    # Create a new input layer that accommodates the concatenated input tensor
    new_input = Conv2D(3, (1, 1), activation='relu')(concatenated_input)
    new_input = BatchNormalization()(new_input)

    # loading pretrained conv base model
    conv_base = efn.EfficientNetB5(weights='imagenet', include_top=False)

    # Use the base model as a layer in your custom model
    conv_base_modify = conv_base(new_input)

    # x = conv_base.output
    # # create new model with a new classification layer
    x = layers.GlobalAveragePooling2D(name = 'head_pooling')(conv_base_modify)
    x = layers.Dropout(0.50, name = 'head_dropout')(x)
    x = layers.Dense(1, activation='sigmoid',name = 'prediction_layer')(x)
    # x = x(conv_base_modify)
    model = models.Model(inputs=[input_img1, input_img2], outputs=x)

    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))

    if fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for layer in conv_base.layers:
            layer.trainable = False

    print('This is the number of trainable layers '
           'after freezing the conv base:', len(model.trainable_weights))
    print('-'*125)

    return input_shape, model





def loadresumemodel(model_dir):
    model = load_model(model_dir)
    height = width = model.input_shape[0][1]  ## model.input_shape[0][1]
    input_shape = (height, width, 3)
    
    return input_shape, model




def model_block4Unfreze(model_dir):
    model = load_model(model_dir)
    input_shape = (model.input_shape[1][1], model.input_shape[1][2], model.input_shape[1][3])
    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.layers[5].trainable_weights))
    layers = model.layers[5].layers
    for innerlayer in layers:
        if innerlayer.name.startswith("block4"):
#             print(innerlayer.name)
            innerlayer.trainable = True
    ## Unfreeze FC layer
    fc_layer = model.get_layer("prediction_layer")
    fc_layer.trainable = True
    
    print('This is the number of trainable layers '
          'after freezing the block5a_se_excite Layer:', len(model.layers[5].trainable_weights))

    return input_shape, model

















