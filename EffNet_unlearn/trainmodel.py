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
from keras.optimizers import Adam
from model import build_modelB5_unlearn, loadresumemodel, model_block4Unfreze, model_block4TOblock7Unfreze, model_block1TOblock4Unfreze
from data_generator import batch_datagen, Flip_generator
#load Check point
from tensorflow.keras.models import load_model
import argparse



def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)



def cal_steps(num_images, batch_size):
    # calculates steps for generator
    steps = num_images // batch_size

   # adds 1 to the generator steps if the steps multiplied by
   # the batch size is less than the total training samples
    return steps + 1 if (steps * batch_size) < num_images else steps




def main():
     # construct the argument parser
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train our network for')
    my_parser.add_argument('--gpu', type=int, default=1, help='Number GPU 0,1')
    #my_parser.add_argument('--data', type=str, default='mini-ImageNet')
    my_parser.add_argument('--data_path', type=str, default='/home/kannika/codes_AI/CSV/mini-ImageNet_MachineUnlearn.csv')
    my_parser.add_argument('--save_dir', type=str, help='Main Output Path', default="/media/HDD/mini-ImageNet/EffNetB5Model_unlearn")
    my_parser.add_argument('--name', type=str, default=".", help='[transfer, unfreezeB4, unfreezeB1-B4, unfreezeB4-B7, unfreezeB1-B4 ]')
    my_parser.add_argument('--R', type=int, help='[1:R1, 2:R2]')
    my_parser.add_argument('--lr', type=float, default=1e-5)
    my_parser.add_argument('--batchsize', type=int, default=16)
    my_parser.add_argument('--resume', action='store_true')
    my_parser.add_argument('--checkpoint_dir', type=str ,default=".")
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
    if R == 2:
        root_base = f'{save_dir}/{_R}/{args.name}'
    else:
        root_base = f'{save_dir}/{_R}'
        
    os.makedirs(root_base, exist_ok=True)
    data_path = args.data_path
    BATCH_SIZE = args.batchsize
    ## train seting
    epochs = args.epochs
    
    ### Create Model 
    if args.resume :
         input_shape, model = loadresumemodel(args.checkpoint_dir)
    elif args.R == 2 and args.name == "unfreezeB4" :
        print("[INFO]: Load Model to Finetune Stage: Train Middle and Fully Connected Layers, and Leave Others Frozen")
        input_shape, model = model_block4Unfreze(args.checkpoint_dir)
    elif args.R == 2 and args.name == "unfreezeB4-B7" :
        print("[INFO]: Load Model to Finetune Stage: Train Middle to the Last Layer and Fully Connected Layer, and Leave Others Frozen")
        input_shape, model = model_block4TOblock7Unfreze(args.checkpoint_dir)
    elif args.R == 2 and args.name == "unfreezeB1-B4" :
        print("[INFO]: Load Model to Finetune Stage: Train Primary to the Middle Layers and Fully Connected Layer, and Leave Others Frozen")
        input_shape, model = model_block1TOblock4Unfreze(args.checkpoint_dir)
    elif args.resume and args.R == 2:
        input_shape, model = loadresumemodel(args.checkpoint_dir)
        print(" ==================================== [INFO]: Resume Model to Finetune Stage ====================================")
    else:    
        input_shape, model = build_modelB5_unlearn(fine_tune=True)
    ##get images size 
    IMAGE_SIZE = input_shape[0]
    model.summary()
    print('='*100)
    
    ## import dataset
    dataset = pd.read_csv(data_path, dtype=str)
    Train_df = dataset[dataset['subset']=='train'].reset_index(drop=True)        
    val_df = dataset[dataset['subset']=='val'].reset_index(drop=True)
    ### Get data Loder  ## input_shape ==> (456, 456, 3)
    ### Implement > ## Train set  
    batch_generator_main, batch_generator_2 = batch_datagen(Train_df, input_shape, BATCH_SIZE=args.batchsize)
    batch_train = Flip_generator(batch_generator_main, batch_generator_2, IMAGE_SIZE=IMAGE_SIZE)
    ## Validation set 
    batchval_generator_main, batchval_generator_2 = batch_datagen(val_df, input_shape, BATCH_SIZE=args.batchsize)
    batch_val = Flip_generator(batchval_generator_main, batchval_generator_2, IMAGE_SIZE=IMAGE_SIZE)

    ## Set mkdir TensorBoard 
    ##root_logdir = f'/media/SSD/rheology2023/VitModel/Regression/tensorflow/ExpTest/R1/Mylogs_tensor/'
    root_logdir = f"{root_base}/{args.tensorName}"
    os.makedirs(root_logdir, exist_ok=True)
    ### Run TensorBoard 
    run_logdir = get_run_logdir(root_logdir)
    tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)
    ## Create Model Folder 
    modelNamemkdir = f"{root_base}/{args.FmodelsName}"
    os.makedirs(modelNamemkdir, exist_ok=True)
    ## Create checkpointer Folder 
#     checkpointerdir = f"{root_base}/{args.checkpointerName}"
#     os.makedirs(checkpointerdir, exist_ok=True)
    #checkpoint_path = f'{checkpointerdir}/cp_modelEffNetB5_Unlearning_{args.name}_{_R}_epoch-{epoch:04d}.ckpt'
    ## Set Model Name 
    modelName = f'modelEffNetB5_Unlearning_{args.name}_{_R}.h5'
    ## Set check point Name
    on_epochName = f'modelEffNetB5_Unlearning_{args.name}_{_R}'
    Model2save = f'{modelNamemkdir}/{modelName}'
    root_Metrics = f'{root_base}/{args.epochendName}/'
    os.makedirs(root_Metrics, exist_ok=True)
    class Metrics(Callback):
            def on_epoch_end(self, epochs, logs={}):
                if epochs%50 == 0 and epochs != 0:
                    self.model.save(f'{root_Metrics}{on_epochName}_epoch{epochs}.h5')
                    # Save the weights
                    #self.model.save_weights(f'{checkpointerdir}/my_checkpoint_epoch-{epochs}/cp_modelEffNetB5_Unlearning_{args.name}_{_R}_epoch{epochs}')
                else:
                    self.model.save(f'{root_Metrics}{modelName}')
                    #self.model.save_weights(f'{checkpointerdir}/my_checkpoint_last/cp_modelEffNetB5_Unlearning_{args.name}_{_R}')
                return
    
    ## For tracking Quadratic Weighted Kappa score and saving best weights
    metrics = Metrics()
    ### Set shows train steps:
    steps_per_epoch = cal_steps(len(Train_df), batch_size=args.batchsize)
    validation_steps = cal_steps(len(val_df), batch_size=args.batchsize)
#     # Create a callback that saves the model's weights every 50 epochs
#     cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, 
#                                                          save_weights_only=True, save_freq= int(50*steps_per_epoch))
    ### Set optimizers 
    loss_smooth = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2)
    opt = tf.keras.optimizers.legacy.Adam(args.lr, decay=1e-4)

    #Training
    model.compile(
            optimizer=opt,
            loss=loss_smooth,
            metrics=['accuracy'])

    ## Fit model 
    history = model.fit(batch_train, steps_per_epoch=steps_per_epoch,
                epochs=epochs, batch_size=args.batchsize,
                validation_data=batch_val, validation_steps=validation_steps,
                callbacks = [metrics, tensorboard_cb])
   
    
    # Save model as .h5        
    model.save(Model2save)
    ### print
    print(f"Save USAI Model (EffNetB5 Unlearning) as: {Model2save}")
    print(f"*"*150)
    

    
    
## Run Function 
if __name__ == '__main__':
    main()
    
    






