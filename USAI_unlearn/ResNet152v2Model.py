import tensorflow as tf
from skimage.io import imread
from keras.utils import generic_utils
from keras import layers
from keras import models
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Concatenate, Input, Conv2D, BatchNormalization, Input, GlobalAveragePooling2D, Dropout, Dense
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


def ResNetUnfreeze_conv3_block(path_modelJson, path_modelweights, sets):
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
    if sets == "MLunlearn_USAI":
        print(" ========== [INFO]: Get ResRes152v2 layers to Finetune Stage. ==========")
        resnet_layer = model.layers[1]
        for innerlayer in resnet_layer.layers:
            if innerlayer.name.startswith("conv3_block"):
                innerlayer.trainable = True
    elif sets == "MLorigin_USAI":
        ## Unfreeze conv3_block layers.
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


def ResNetUnfreeze_conv3to5_block(path_modelJson, path_modelweights, sets):
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
           'before freezing the conv3_block - conv5_block layers:', len(model.trainable_weights))
    if sets == "MLunlearn_USAI":
        print(" ========== [INFO]: Get ResRes152v2 layers to Finetune Stage. ==========")
        resnet_layer = model.layers[1]
        resnet_layer.trainable = True
        set_trainable = False
        for innerlayer in resnet_layer.layers:
            if innerlayer.name.startswith('conv3_block'):
                set_trainable = True
            if set_trainable:
                innerlayer.trainable = True
            else:
                innerlayer.trainable = False
    elif sets == "MLorigin_USAI":
        model.trainable = True
        set_trainable = False
        for layer in model.layers:
            if layer.name.startswith('conv3_block'):
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

    print('This is the number of trainable layers '
           'after freezing the conv3_block - conv5_block layers:', len(model.trainable_weights))
    ## Get Model Input Shape 
    input_shape = (model.input_shape[1], model.input_shape[2], model.input_shape[3])

    return input_shape, model


def ResNetUnfreeze_conv1xto3_block(path_modelJson, path_modelweights, sets):
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
           'before freezing the conv1_x - conv3_block layers:', len(model.trainable_weights))
    if sets == "MLunlearn_USAI":
        print(" ========== [INFO]: Get ResRes152v2 layers to Finetune Stage. ==========")
        resnet_layer = model.layers[1]
        for innerlayer in resnet_layer.layers:
            if any(innerlayer.name.startswith(conv) for conv in ["conv1", "conv2", "conv3"]):
                innerlayer.trainable = True
            else:
                innerlayer.trainable = False  # Optional: freeze other layers
    elif sets == "MLorigin_USAI":
        for layer in model.layers:
            if any(layer.name.startswith(conv) for conv in ["conv1", "conv2", "conv3"]):
                layer.trainable = True
            else:
                layer.trainable = False  # Optional: freeze other layers
    ## Unfreeze FC layer
    fc_layer = model.get_layer("prediction_layer")
    fc_layer.trainable = True

    print('This is the number of trainable layers '
           'after freezing the  conv1_x - conv3_block layers:', len(model.trainable_weights))
    ## Get Model Input Shape 
    input_shape = (model.input_shape[1], model.input_shape[2], model.input_shape[3])

    return input_shape, model
  
    
    
'''
Function: Resume Model

'''

def loadresumemodel_ResNet(path_modelJson, path_modelweights):
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
    ## Get Model Input Shape 
    input_shape = (model.input_shape[1], model.input_shape[2], model.input_shape[3])
    print("[INFO]: Loaded model from disk")
    print('[INFO]: This is the number of trainable layers: ', len(model.trainable_weights))
    
    return input_shape, model





'''
#################################################################################################
Create ResNet152v2 Unlearn model for Train on USAI 15AB Dataset: Re-adjust input to single input
#################################################################################################
'''

def CreateResNet152v2modelUnlearn(path_modelJson, path_modelweights, imagesize, Numclasses=15):
    '''
    param: path/to/model/Json/file.
         :path/to/model/weights/file.
    '''
    
    json_file = open(path_modelJson, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    old_model = model_from_json(loaded_model_json)
    #load weights into new model  -------------------
    old_model.load_weights(path_modelweights)
    print("Loaded model from disk")
    # Get ResNet152v2 model
    ResNet152v2_layer = old_model.get_layer('resnet152v2')  # 
    ResNet152v2_layer.trainable =  False #
    # Step 3: สร้าง input layer ใหม่ที่มีขนาด (224, 224, 3)
    height=width=imagesize
    new_input = Input(shape=(height, width, 3), name='new_input')
    input_shape = (height, width, 3)
    # Step 4: ต่อเลเยอร์ EResNet152v2_layer เข้ากับ input ใหม่
    x = ResNet152v2_layer(new_input)
    # Step 5: เพิ่มเลเยอร์ Global Average Pooling, Dropout, และเลเยอร์ output (15 classes)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # สามารถปรับ dropout ตามความเหมาะสม
    output = Dense(Numclasses, activation='softmax', name='prediction_layer')(x)  # 15 classes ใช้ softmax สำหรับ multi-class classification
    # Step 6: สร้างโมเดลใหม่
    new_model = Model(inputs=new_input, outputs=output, name="ResNet152v2_unlearnUSAI15ab")
    # Step 7: Compile โมเดลใหม่ (ปรับ optimizer และ loss function ตามที่ต้องการ)
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #showing before&after freezing
    print('This is the number of trainable layers '
          'before freezing the conv base:', len(new_model.trainable_weights))
    #conv_base.trainable = False  # freeze เพื่อรักษา convolutional base's weight
    layers = new_model.layers[1].layers
    for innerlayer in layers:
        innerlayer.trainable = False
    ### Ensure to Unfreeze FC layer
    fc_layer = new_model.get_layer("prediction_layer")
    fc_layer.trainable = True
    print('This is the number of trainable layers '
          'after freezing the conv base:', len(new_model.trainable_weights))

    return input_shape, new_model

