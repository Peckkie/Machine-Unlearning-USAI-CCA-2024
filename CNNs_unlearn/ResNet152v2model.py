import tensorflow as tf
from skimage.io import imread
from keras.utils import generic_utils
from keras import layers
from keras import models
from tensorflow.keras import optimizers
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Concatenate, Input, Conv2D, BatchNormalization
#load Check point
from tensorflow.keras.models import load_model, Model, model_from_json


'''
Function: Create ResNet152v2 model pair-two images input to train ML Unlearn with miniImageNet dataset.
'''

def build_ResNet152v2_unlearn(imagesize):
    """
    :param: imagesize ==> forResNet152v2 = 224x224 pixel
    """

    # Define the input shapes for the two images
    img_height, img_width, channels = imagesize, imagesize, 3
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
    conv_base = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False)
    # Use the base model as a layer in your custom model
    conv_base_modify = conv_base(new_input)
    # x = conv_base.output
    # # create new model with a new classification layer
    x = layers.GlobalAveragePooling2D(name = 'head_pooling')(conv_base_modify)
    x = layers.Dropout(0.50, name = 'head_dropout')(x)
    x = layers.Dense(1, activation='sigmoid',name = 'prediction_layer')(x)
    # x = x(conv_base_modify)
    model = models.Model(inputs=[input_img1, input_img2], outputs=x, name="ResNet152v2-Unlearn")

    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))

    print('[INFO]: Freezing hidden layers...')
    for layer in conv_base.layers:
        layer.trainable = False

    print('This is the number of trainable layers '
           'after freezing the conv base:', len(model.trainable_weights))
    print('-'*125)

    return input_shape, model



def resumeMOdelResNet152v2(path_modelJson, path_modelweights):
    json_file = open(path_modelJson, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #load weights into new model  -------------------
    model.load_weights(path_modelweights)
    print("Loaded model from disk")
    getshape = model.input_shape
    input_shape = (getshape[0][1], getshape[0][2], getshape[0][3]) 

    return input_shape, model


def ResNet152v2Unfreeze_conv3_block(path_modelJson, path_modelweights):
    json_file = open(path_modelJson, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #load weights into new model  -------------------
    model.load_weights(path_modelweights)
    print("Loaded model from disk")
    getshape = model.input_shape
    input_shape = (getshape[0][1], getshape[0][2], getshape[0][3]) 
    ### Get ResNet152v2 model layers
    resnet_layers = model.layers[5]
    print('[INFO]: This is the number of trainable ResNet152v2 layers '
              'before unfreeze the conv3_block layers:', len(resnet_layers.trainable_weights))
    for innerlayer in resnet_layers.layers:
        if innerlayer.name.startswith("conv3_block"):
            innerlayer.trainable = True
    ## Unfreeze FC layer
    fc_layer = model.get_layer("prediction_layer")
    fc_layer.trainable = True
    print('[INFO]: This is the number of trainable ResNet152v2 layers '
              'after unfreeze the conv3_block layers:', len(resnet_layers.trainable_weights))

    return input_shape, model


def ResNet152v2Unfreeze_conv3_blockTOconv5_block(path_modelJson, path_modelweights):
    json_file = open(path_modelJson, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #load weights into new model  -------------------
    model.load_weights(path_modelweights)
    print("Loaded model from disk")
    getshape = model.input_shape
    input_shape = (getshape[0][1], getshape[0][2], getshape[0][3]) 
    ### Get ResNet152v2 model layers
    resnet_layers = model.layers[5]
    print('[INFO]: This is the number of trainable ResNet152v2 layers '
              'before unfreeze the conv3_block - conv5_block layers:', len(resnet_layers.trainable_weights))
    resnet_layers.trainable = True
    set_trainable = False
    for layer in resnet_layers.layers:
        if layer.name.startswith('conv3_block'):
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    ## Unfreeze FC layer
    fc_layer = model.get_layer("prediction_layer")
    fc_layer.trainable = True
    print('[INFO]: This is the number of trainable ResNet152v2 layers '
              'after unfreeze the conv3_block - conv5_block layers:', len(resnet_layers.trainable_weights))

    return input_shape, model


def ResNet152v2Unfreeze_conv1xTOconv3_block(path_modelJson, path_modelweights):
    json_file = open(path_modelJson, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #load weights into new model  -------------------
    model.load_weights(path_modelweights)
    print("Loaded model from disk")
    getshape = model.input_shape
    input_shape = (getshape[0][1], getshape[0][2], getshape[0][3]) 
    ### Get ResNet152v2 model layers
    resnet_layers = model.layers[5]
    print('[INFO]: This is the number of trainable ResNet152v2 layers '
              'before unfreeze the conv1x - conv3_block layers:', len(resnet_layers.trainable_weights))
    for innerlayer in resnet_layers.layers:
        if any(innerlayer.name.startswith(conv) for conv in ["conv1", "conv2", "conv3"]):
            innerlayer.trainable = True
        else:
            innerlayer.trainable = False  # Optional: freeze other layers
    ## Unfreeze FC layer
    fc_layer = model.get_layer("prediction_layer")
    fc_layer.trainable = True
    print('[INFO]: This is the number of trainable ResNet152v2 layers '
              'after unfreeze the conv1x - conv3_block layers:', len(resnet_layers.trainable_weights))

    return input_shape, model



