import os
import tensorflow as tf
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import Callback, TensorBoard
import pandas as pd
from keras.utils import generic_utils
from keras import layers
from keras import models
from tensorflow.keras import optimizers
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from EffNetmodel import build_modelB5_unlearn, loadresumemodel, model_block4Unfreze, model_block4TOblock7Unfreze, model_block1TOblock4Unfreze
from EffNetmodel import model_block5a_se_excite_Unfreze, build_modelB5_unlearn
from ResNet152v2model import build_ResNet152v2_unlearn, resumeMOdelResNet152v2, ResNet152v2Unfreeze_conv3_block, ResNet152v2Unfreeze_conv3_blockTOconv5_block, ResNet152v2Unfreeze_conv1xTOconv3_block
from data_generator import batch_datagen, Flip_generator
#load Check point
from tensorflow.keras.models import load_model
import argparse



'''
Function: run tensorboard for tracking accuracy and loss.
'''
def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)


'''
Function: Callback for save model every epoch and 50 epochs.
'''
class Metrics(Callback):
    def __init__(self, root_metrics, on_epoch_name):
        super(Metrics, self).__init__()
        self.root_metrics = root_metrics
        self.on_epoch_name = on_epoch_name

    def on_epoch_end(self, epochs, logs=None):
        if logs is None:
            logs = {}
        # Check if this is a checkpoint epoch (every 50 epochs)
        if epochs % 20 == 0 and epochs != 0:
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
                

'''
Function: Calculate step number for each epoch for training model unlearning.
'''
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
    my_parser.add_argument('--gpu', type=int, default=1, help='Number GPU 0,1 [Server 28, device: 1, name: NVIDIA GeForce RTX 3090 Ti]')
    my_parser.add_argument('--network_name', type=str, default='ResNet152v2', help='[ResNet152v2, EffNetB5]')
    my_parser.add_argument('--set', type=str, default='baseML_unlearn', help='In this work only [baseML_unlearn], For finetuning with USAI 15 AB [MLunlearn_USAI, MLorigin_USAI]')
    my_parser.add_argument('--data_path', type=str, default='/home/kannika/code/mini-ImageNet_MachineUnlearn.csv', help='Path to CSV file.')
    my_parser.add_argument('--save_dir', type=str, default="/media/tohn/HDD2/Model_unlearn", help='Main Output Path >> [/media/HDD/mini-ImageNet/{network_name}Model]')
    my_parser.add_argument('--imgsize', type=int, default=224, help='[EffNetB5: 456, ResNet152v2: 224]')
    my_parser.add_argument('--name', type=str, default=".", help='EffNetB5: [transfer, unfreezeB4, unfreezeB1-B4, unfreezeB4-B7, unfreezeB1-B4, unfreezeBlock5a_se_excite], ResNet152v2: [transfer, unfreeze_conv3_block, unfreeze_conv3_block-conv5_block, unfreeze_conv1-conv3_block]')
    my_parser.add_argument('--R', type=int, help='[1:R1, 2:R2]')
    my_parser.add_argument('--lr', type=float, default=1e-5, help='[Train Unlearn: 1e-5]')
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
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}" 
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices))

    ## Create Model
    if args.network_name == "EffNetB5":
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
        elif args.R == 2 and args.name == "unfreezeB5a_se_excite" :
            print("[INFO]: Load Model to Finetune Stage: Train Primary to the Block5a Layers and Fully Connected Layer, and Leave Others Frozen")
            input_shape, model = model_block5a_se_excite_Unfreze(args.checkpoint_dir)
        elif args.resume and args.R == 2:
            input_shape, model = loadresumemodel(args.checkpoint_dir)
            print(" ==================================== [INFO]: Resume Model to Finetune Stage ====================================")
        else: 
            print(" ==================================== [INFO]: Build EffNetB5 model unlearning Stage ====================================")
            input_shape, model = build_modelB5_unlearn(fine_tune=True)
    elif args.network_name == "ResNet152v2":
        if args.R == 1 and args.name == "transfer" :
            print(" ================================ [INFO]: Build ResNet152v2 model unlearning Stage ================================")
            input_shape, model = build_ResNet152v2_unlearn(imagesize=args.imgsize) 
        elif args.R == 2 and args.name == "unfreeze_conv3_block" :
            print("[INFO]: Load ResNet152v2 Unlearn Model to Finetune Stage: Train conv3_block and Fully Connected Layers")
            input_shape, model = ResNet152v2Unfreeze_conv3_block(args.Modeljson_dir, args.checkpoint_dir)
        elif args.R == 2 and args.name == "unfreeze_conv3_block-conv5_block" :
            print("[INFO]: Load ResNet152v2 Unlearn Model to Finetune Stage: Train conv3_block TO conv5_block and Fully Connected Layers")
            input_shape, model = ResNet152v2Unfreeze_conv3_blockTOconv5_block(args.Modeljson_dir, args.checkpoint_dir)
        elif args.R == 2 and args.name == "unfreeze_conv1-conv3_block" :
            print("[INFO]: Load ResNet152v2 Unlearn Model to Finetune Stage: Train conv1x TO conv3_block and Fully Connected Layers")
            input_shape, model = ResNet152v2Unfreeze_conv1xTOconv3_block(args.Modeljson_dir, args.checkpoint_dir)
        elif args.resume :
            print(f"[INFO]: ================== [INFO]: Load ResNet152v2 Unlearn Model to Resume Train R{args.R} Stage ==================")
            input_shape, model = resumeMOdelResNet152v2(args.Modeljson_dir, args.checkpoint_dir)
    ## get images size 
    IMAGE_SIZE = input_shape[0]
    ### summary model
    model.summary()
    print('='*100)

    ## Create Data Training Loader
    dataset = pd.read_csv(args.data_path, dtype=str)
    Train_df = dataset[dataset['subset']=='train'].reset_index(drop=True)        
    val_df = dataset[dataset['subset']=='val'].reset_index(drop=True)
    ### Implement > ## Train set  
    batch_generator_main, batch_generator_2 = batch_datagen(Train_df, input_shape, BATCH_SIZE=args.batchsize)
    batch_train = Flip_generator(batch_generator_main, batch_generator_2, IMAGE_SIZE=IMAGE_SIZE)
    ## Validation set 
    batchval_generator_main, batchval_generator_2 = batch_datagen(val_df, input_shape, BATCH_SIZE=args.batchsize)
    batch_val = Flip_generator(batchval_generator_main, batchval_generator_2, IMAGE_SIZE=IMAGE_SIZE)

    ## Create Main Output Path
    _R = f"R{args.R}"
    root_base = f'{args.save_dir}/{args.network_name}Model/{args.set}/{_R}/{args.name}'
    os.makedirs(root_base, exist_ok=True)
    ## Set mkdir TensorBoard 
    root_logdir = f"{root_base}/{args.tensorName}"
    os.makedirs(root_logdir, exist_ok=True)
    ### Run TensorBoard 
    run_logdir = get_run_logdir(root_logdir)
    #tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)
    tensorboard_cb = TensorBoard(log_dir=run_logdir, write_graph=False)
    ## Create Models Name 
    modelName = f'model{args.network_name}_Unlearning_miniImageNet_{args.name}-{_R}.h5'
    on_epochName = f'model{args.network_name}_Unlearning_miniImageNet_{args.name}-{_R}'
    ## Create Model Folder 
    modelNamemkdir = f"{root_base}/{args.FmodelsName}"
    os.makedirs(modelNamemkdir, exist_ok=True)
    Model2save = f'{modelNamemkdir}/{modelName}'
    ## Create save epoch end folder  
    root_Metrics = f'{root_base}/{args.epochendName}/'
    os.makedirs(root_Metrics, exist_ok=True)
    ## Initialize the Metrics callback
    metrics_callback = Metrics(root_metrics=root_Metrics, on_epoch_name=on_epochName)
    ## Setting model complier
    ### Set shows train steps:
    steps_per_epoch = cal_steps(len(Train_df), batch_size=args.batchsize)
    validation_steps = cal_steps(len(val_df), batch_size=args.batchsize)
    ### Set loss and opimizer 
    loss_smooth = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=args.lr, decay=1e-4)
    #Compliermodel
    model.compile(optimizer=opt, loss=loss_smooth, metrics=['accuracy'])
    # Fit model 
    model.fit(batch_train, steps_per_epoch=steps_per_epoch, epochs=args.epochs, batch_size=args.batchsize,
                validation_data=batch_val, validation_steps=validation_steps,
                callbacks = [metrics_callback, tensorboard_cb])


    ### Save models
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
    
    






