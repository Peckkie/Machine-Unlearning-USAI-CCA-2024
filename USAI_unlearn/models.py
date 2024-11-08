import tensorflow as tf
from keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from efficientnet.keras import center_crop_and_resize, preprocess_input
import tensorflow.keras as keras
from keras import models
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense#, Conv2D, BatchNormalization
from tensorflow.keras.applications import EfficientNetB5 as Net



def loadresumemodel(path_model):
    model = load_model(path_model)
    height = width = model.input_shape[1]
    input_shape = (height, width, 3)
    
    return input_shape, model


def loadmodelUnlearn(path_model):
    # Step 1: โหลดโมเดลเดิมที่มี EfficientNet-B5
    old_model = load_model(path_model)
    # Step 2: ดึง EfficientNet-B5 จากโมเดลเดิม
    efficientnet_b5_layer = old_model.get_layer('efficientnet-b5')  # ดึงเฉพาะ EfficientNet-B5
    efficientnet_b5_layer.trainable =  False #True  # สามารถเลือกให้ฝึกต่อได้ หรือจะ freeze ก็ได้
    # Step 3: สร้าง input layer ใหม่ที่มีขนาด (456, 456, 3)
    height=width = 456
    new_input = Input(shape=(height, width, 3), name='new_input')
    input_shape = (height, width, 3)
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


def finetuneUSAI_B4(path_modelJson, path_modelweights):
    ##load json and create model --------------------
    json_file = open(path_modelJson, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #load weights into new model  -------------------
    model.load_weights(path_modelweights)
    print("Loaded model from disk")
    input_shape = (model.input_shape[1], model.input_shape[2], model.input_shape[3])
    ### Unfreeze Block4 
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
    
    return input_shape, model


def build_EffNetmodelB5(fine_tune, Numclasses):
    
    """
    :param fine_tune (bool): Whether to train the hidden layers or not.
    :Numclasses == 15 AB ==> SubClass New 
    """
    height = width = 456
    input_shape = (height, width, 3)
    # loading pretrained conv base model
    conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape)
    # create new model with a new classification layer
    x = conv_base.output  
    global_average_layer = layers.GlobalAveragePooling2D(name = 'head_pooling')(x)
    dropout_layer_1 = layers.Dropout(0.50,name = 'head_dropout')(global_average_layer)
    prediction_layer = layers.Dense(Numclasses, activation='softmax',name = 'prediction_layer')(dropout_layer_1)
    ## Create model 
    model = models.Model(inputs= conv_base.input, outputs=prediction_layer) 
    ### Unfreeze FC layers 
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


def model_block5Unfreze(path_modelJson, path_modelweights):
    ##load json and create model --------------------
    json_file = open(path_modelJson, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #load weights into new model  -------------------
    model.load_weights(path_modelweights)
    #get input shape
    height = width = model.input_shape[1]
    input_shape = (height, width, 3)
    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))
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
          'after freezing the block5a_se_excite Layer:', len(model.trainable_weights))

    return input_shape, model
    
    
    
    
    
    
    
    
    
    
    



