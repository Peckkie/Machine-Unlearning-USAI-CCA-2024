import tensorflow as tf
from skimage.io import imread
from keras.utils import generic_utils
from keras import layers
from keras import models
from tensorflow.keras import optimizers
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Concatenate, Dense, Input, Conv2D, BatchNormalization
from vit_keras import vit
#load Check point
from tensorflow.keras.models import load_model

'''
Function: Create ViT model for 2 Inputs 
'''

def build_vit_unlearn(image_size):
    input1 = Input(shape=(image_size, image_size, 3), name="input1")
    input2 = Input(shape=(image_size, image_size, 3), name="input2")
    # Concatenate the two inputs along the channel dimension
    combined_input = Concatenate(axis=-1)([input1, input2])  # Shape: (batch_size, image_size, image_size, 6)
    # Reduce the channels to 3 using a Conv2D layer
    reduced_input = Conv2D(3, kernel_size=1, activation="gelu", name="reduce_channels")(combined_input)
    normalized_input = BatchNormalization(name="batch_norm")(reduced_input)
    # Pass the combined input to the ViT model
    vit_model = vit.vit_l32(image_size=image_size, activation='sigmoid', pretrained=True,
                                include_top=False, pretrained_top=False)
    vit_output = vit_model(normalized_input)
    ### add the tail layer ###  
    Flatten_layer = layers.Flatten()(vit_output)
    BatchNormalization_layer1 = layers.BatchNormalization(name='BatchNormalization_1')(Flatten_layer)
    Dense_layer1 = layers.Dense(256, activation='relu',name='Dense256')(BatchNormalization_layer1)
    BatchNormalization_layer2 = layers.BatchNormalization(name='BatchNormalization_2')(Dense_layer1)
    output = layers.Dense(1, activation='sigmoid',name='pred_layer')(BatchNormalization_layer2)
    # Define the full model
    model = tf.keras.Model(inputs=[input1, input2], outputs=output, name = 'Vit_Unlearn')
    fc_layer = model.get_layer('pred_layer')         
    fc_layer.trainable = True
    print('This is the number of trainable layers: ', len(model.trainable_weights))
    input_shape = (model.input_shape[1][1], model.input_shape[1][2], model.input_shape[1][3])

    
    return input_shape, model


def loadresumemodel(model_dir):
    model = load_model(model_dir)
    height = width = model.input_shape[0][1]  ## model.input_shape[0][1]
    input_shape = (height, width, 3)
    
    return input_shape, model


'''
Function: Create ViT model for train USAI15AB
'''

def build_ViTL32Model(IMAGE_SIZE, Numclass):
    """
    :param 
    IMAGE_SIZE: 384x384 pixel for ViT L-32 patchs.  
    Numclass: Number of Class to train ; 15 => 15 Classes USAI image. 
    """
    
    vit_model = vit.vit_l32(image_size=IMAGE_SIZE, classes=Numclass, activation = 'softmax', pretrained = True, 
                            include_top = False, pretrained_top = False)
    print('[INFO]: Loading pre-trained weights')
    x = vit_model.get_layer('ExtractToken').output
    ### add the tail layer ###  
    Flatten_layer1 = layers.Flatten()(x)
    BatchNormalization_layer1 = layers.BatchNormalization(name='BatchNormalization_1')(Flatten_layer1)
    Dense_layer1 = layers.Dense(256, activation='gelu',name='Dens256')(BatchNormalization_layer1)
    BatchNormalization_layer2 = layers.BatchNormalization(name='BatchNormalization_2')(Dense_layer1)
    Dense_layer2 = layers.Dense(Numclass, activation='softmax',name='Pred_15AB')(BatchNormalization_layer2)
        
    model = models.Model(inputs= vit_model.input, outputs=[Dense_layer2], name = 'ViT_USAI') 
    ### Unfreeze FC Layer 
    print('This is the number of trainable layers before freezing the conv base:', len(model.trainable_weights))
    fc_layer = model.get_layer('Pred_15AB')
    fc_layer.trainable = True
    print('This is the number of trainable layers: ', len(model.trainable_weights))
    ### Get input shape
    height = width = model.input_shape[1]
    input_shape = (height, width, 3)
    print(input_shape)


    return input_shape, model



def loadresumemodel_1input(model_dir):
    model = load_model(model_dir)
    height = width = model.input_shape[1]  ## model.input_shape[0][1]
    input_shape = (height, width, 3)
    
    return input_shape, model

