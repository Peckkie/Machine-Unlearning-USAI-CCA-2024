import tensorflow as tf
from skimage.io import imread
from keras.utils import generic_utils
from keras import layers
from keras import models
from tensorflow.keras.models import model_from_json
from tensorflow.keras import optimizers
from keras.optimizers import Adam
from tensorflow.keras.layers import Concatenate, Input, Conv2D, BatchNormalization
#load Check point
from tensorflow.keras.models import load_model


'''
Function: Create ResNet152v2 Base model to train on USAI 15AB dataset.
'''

def build_baseResNet152v2(imagesize, Numclasses=15):
    
    """
    :param: imagesize ==> forResNet152v2 = 224x224 pixel
    :Numclasses == 15 AB ==> SubClass New. 
    """
    height = width = imagesize
    input_shape = (height, width, 3)
    # loading pretrained conv base model
    conv_base = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, 
                                                         input_shape=input_shape)
    # create new model with a new classification layer
    x = conv_base.output  
    global_average_layer = layers.GlobalAveragePooling2D(name = 'head_pooling')(x)
    dropout_layer_1 = layers.Dropout(0.50,name = 'head_dropout')(global_average_layer)
    prediction_layer = layers.Dense(Numclasses, activation='softmax',name = 'prediction_layer')(dropout_layer_1)
    ## Create model 
    model = models.Model(inputs= conv_base.input, outputs=prediction_layer, name="ResNet152v2_USAI15AB") 
    ### Unfreeze FC layers 
    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))

    print('[INFO]: Freezing Conv. layers...')
    for layer in conv_base.layers:
        layer.trainable = False

    print('This is the number of trainable layers '
           'after freezing the conv base:', len(model.trainable_weights))
    print('-'*125)

    return input_shape, model


def ResNetUnfreeze_conv3_block(path_modelJson, path_modelweights):
    '''
    param: path/to/model/Json/file.
         :path/to/model/weights/file.
    '''
    json_file = open(path_modelJson, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #load weights into new model  -------------------
    model.load_weights(path_modelweights)
    print("Loaded model from disk")
    print('This is the number of trainable layers '
          'before freezing the conv3_block layers:', len(model.trainable_weights))
    for innerlayer in model.layers:
        if innerlayer.name.startswith("conv3_block"):
            innerlayer.trainable = True
    ## Unfreeze FC layer
    fc_layer = model.get_layer("prediction_layer")
    fc_layer.trainable = True

    print('This is the number of trainable layers '
              'after freezing the conv3_block layers:', len(model.trainable_weights))
    ## Get Model Input Shape 
    input_shape = (model.input_shape[1], model.input_shape[2], model.input_shape[3])

    return input_shape, model
    
