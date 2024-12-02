import os
import tensorflow as tf
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
from tensorflow.keras import callbacks
from keras.callbacks import Callback
import pandas as pd
from keras.utils import generic_utils
from keras import layers
from keras import models
from tensorflow.keras import optimizers
from models import loadresumemodel, loadmodelUnlearn, finetuneUSAI_B4, build_EffNetmodelB5, model_block5Unfreze, finetuneUSAI_B4ToB7
from data_loader import Data_generator
#from efficientnet.keras import EfficientNetB5 as Net
#load Check point
from tensorflow.keras.models import load_model
import argparse



def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)


class Metrics(Callback):
    def __init__(self, root_metrics, on_epoch_name):
        super(Metrics, self).__init__()
        self.root_metrics = root_metrics
        self.on_epoch_name = on_epoch_name

    def on_epoch_end(self, epochs, logs=None):
        if logs is None:
            logs = {}
        # Check if this is a checkpoint epoch (every 50 epochs)
        if epochs % 50 == 0 and epochs != 0:
            # Save model, weights, and JSON at the checkpoint epoch
            self.model.save(f'{self.root_metrics}{self.on_epoch_name}_epoch{epochs}.h5')
            self.model.save_weights(f'{self.root_metrics}{self.on_epoch_name}_epoch{epochs}.weights.h5')
            # Serialize model to JSON
            model_json = self.model.to_json()
            with open(f"{self.root_metrics}{self.on_epoch_name}_epoch{epochs}.json", "w") as json_file:
                json_file.write(model_json)
        
        else:
            # Save the latest model and weights after each epoch
            self.model.save(f'{self.root_metrics}{self.on_epoch_name}_last.h5')
            self.model.save_weights(f'{self.root_metrics}{self.on_epoch_name}_last.weights.h5')
            # Serialize the latest model to JSON
            model_json = self.model.to_json()
            with open(f"{self.root_metrics}{self.on_epoch_name}_last.json", "w") as json_file:
                json_file.write(model_json)





def main():
     # construct the argument parser
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train our network for')
    my_parser.add_argument('--gpu', type=int, default=1, help='Number GPU 0,1')
    my_parser.add_argument('--set', type=str, default='ML-unlearn', help='[ML-unlearn, EffNet-base]')
    my_parser.add_argument('--data_path', type=str, default='/media/tohn/HDD/VISION_dataset/')
    my_parser.add_argument('--save_dir', type=str, help='Main Output Path', default="/media/tohn/HDD/Model_unlearn/Models_USAI")
    my_parser.add_argument('--name', type=str, default=".", help='[transfer, unfreezeB4, unfreezeB1-B4, unfreezeB4-B7, unfreezeB1-B4, Block5a_se_excite]')
    my_parser.add_argument('--R', type=int, help='[1:R1, 2:R2]')
    my_parser.add_argument('--lr', type=float, default=2e-5)
    my_parser.add_argument('--batchsize', type=int, default=16)
    my_parser.add_argument('--resume', action='store_true')
    my_parser.add_argument('--checkpoint_dir', type=str ,default=".")
    my_parser.add_argument('--Modeljson_dir', type=str ,default=".")
    my_parser.add_argument('--tensorName', type=str ,default="Mylogs_tensor")
    #my_parser.add_argument('--checkpointerName', type=str ,default="checkpointer")
    my_parser.add_argument('--epochendName', type=str ,default="on_epoch_end")
    my_parser.add_argument('--FmodelsName', type=str ,default="models")
    
    args = my_parser.parse_args()
    
    ## set gpu
    gpu = args.gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}" 
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices))
    
    ## get my_parser
    save_dir = args.save_dir
    R = args.R
    _R = f'R{R}'
    ## train seting
    epochs = args.epochs
    
    if args.set == "ML-unlearn": 
        root_base = f'{save_dir}/{args.set}/{_R}/{args.name}'
        os.makedirs(root_base, exist_ok=True)
        ## Create Model
        if args.resume :
            input_shape, model = loadresumemodel(args.checkpoint_dir)
        elif args.R == 1 and args.name == "unfreezeB4" :
            print("[INFO]: Load ML Unlearn unfreezeB4 Model to Finetune Stage: Unfreeze FC Layers")
            input_shape, model = loadmodelUnlearn(args.checkpoint_dir)
        elif args.R == 2 and args.name == "unfreezeB4" :
            print("[INFO]: Load ML Unlearn unfreezeB4 Model to Finetune Stage: Unfreeze Block4")
            input_shape, model = finetuneUSAI_B4(args.Modeljson_dir, args.checkpoint_dir)
        elif args.R == 1 and args.name == "unfreezeB4-B7" :
            print("[INFO]: Load ML Unlearn unfreezeB4-B7 Model to Finetune Stage: Unfreeze FC Layers")
            input_shape, model = loadmodelUnlearn(args.checkpoint_dir)
        elif args.R == 2 and args.name == "unfreezeB4-B7" :
            print("[INFO]: Load ML Unlearn unfreezeB4-B7 Model to Finetune Stage: Unfreeze Block4 to Block7")
            input_shape, model = finetuneUSAI_B4ToB7(args.Modeljson_dir, args.checkpoint_dir)
    elif args.set == "EffNet-base": 
        root_base = f'{save_dir}/{_R}'
        os.makedirs(root_base, exist_ok=True)
        ## Create Model
        if args.resume :
            input_shape, model = loadresumemodel(args.checkpoint_dir)
        elif args.R == 1 and args.name == "transfer" :
            print("[INFO]: Build EffNetB5 Base Model to Transfer Learning Stage")
            input_shape, model = build_EffNetmodelB5(fine_tune=True, Numclasses=15)
        elif args.R == 2 and args.name == "Block5a_se_excite" :
            print("[INFO]: Load EffNetB5 Model to Finetune Stage: Unfreeze Block5a_se_excite Layer")
            input_shape, model = model_block5Unfreze(args.Modeljson_dir, args.checkpoint_dir)
    ##get images size 
    IMAGE_SIZE = input_shape[0]
    model.summary()
    print('='*100)
    
    ## import dataset
    dataframe = pd.read_csv(f'{args.data_path}/Traindf_fold4_8_v1.csv')
    #validation
    valframe = pd.read_csv(f'{args.data_path}/Valdf_fold3_v1.csv')
    ### Get data Loader  ## input_shape ==> (456, 456, 3)
    ### Implement > ## Train set  
    train_generator, valid_generator = Data_generator(height=IMAGE_SIZE, width=IMAGE_SIZE, BATCH_SIZE=args.batchsize, 
                                                          dataframe=dataframe, valframe=valframe)

    ## Set mkdir TensorBoard 
    root_logdir = f"{root_base}/{args.tensorName}"
    os.makedirs(root_logdir, exist_ok=True)
    ### Run TensorBoard 
    run_logdir = get_run_logdir(root_logdir)
    tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)
    ## Create Model Folder 
    modelNamemkdir = f"{root_base}/{args.FmodelsName}"
    os.makedirs(modelNamemkdir, exist_ok=True)

    ## Set Model Name 
    modelName = f'modelEffNetB5_{args.set}_{args.name}-{_R}.h5'
    ## Set check point Name
    on_epochName = f'modelEffNetB5_{args.set}_{args.name}-{_R}'
    Model2save = f'{modelNamemkdir}/{modelName}'
    root_Metrics = f'{root_base}/{args.epochendName}/'
    os.makedirs(root_Metrics, exist_ok=True)
    
    # Initialize the Metrics callback
    metrics_callback = Metrics(root_metrics=root_Metrics, on_epoch_name=on_epochName)
    
    #Compile model
    #Training
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=args.lr),
              metrics=['accuracy'])

    ## Fit model 
    model.fit(train_generator, epochs=epochs, batch_size=args.batchsize, 
                validation_data=valid_generator,
                 callbacks = [metrics_callback, tensorboard_cb])
    
    # serialize model to JSON 
    model_json = model.to_json()
    with open(f"{modelNamemkdir}/{on_epochName}.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f"{modelNamemkdir}/{on_epochName}.weights.h5")
    print("Saved model to disk")
    ### Save model to .h5 
    model.save(Model2save)
    ### print
    print(f"Save USAI Model as: {modelNamemkdir}")
    print(f"*"*150)
    

    
    
## Run Function 
if __name__ == '__main__':
    main()
    
    





