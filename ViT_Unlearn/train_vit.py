import os
import tensorflow as tf
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras import layers, models, optimizers
import pandas as pd
from keras.utils import generic_utils
from keras import layers
from keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam, RMSprop
from vit_model import loadresumemodel, build_vit_unlearn, loadresumemodel_1input, build_ViTL32Model
from vit_dataloader import batch_datagen, Flip_generator, Data_generator
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
                

'''
Function: Calculate step number for each epoch for training model unlearning.
'''
def cal_steps(num_images, batch_size):
    # calculates steps for generator
    steps = num_images // batch_size
   # adds 1 to the generator steps if the steps multiplied by
   # the batch size is less than the total training samples
    return steps + 1 if (steps * batch_size) < num_images else steps


'''
Function: avoid error during training.
'''
def avoid_error(gen):
     while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass


        
def main():
     # construct the argument parser
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train our network for')
    my_parser.add_argument('--gpu', type=int, default=1, help='Number GPU 0,1 [device on 28: 1, name: NVIDIA GeForce RTX 3090 Ti]')
    my_parser.add_argument('--set', type=str, default='ViT_unlearn', help='[ViT_unlearn, ViTunlearn_USAI, ViTorigin_USAI]')
    my_parser.add_argument('--data_path', type=str, default='/media/tohn/HDD/VISION_dataset', help='Path to CSV >> [miniImageNet: /home/kannika/codes_AI/CSV/mini-ImageNet_MachineUnlearn.csv], [USAI: /media/tohn/HDD/VISION_dataset]')
    my_parser.add_argument('--data', type=str, default=".", help='For train with USAI15ab dataset [balanced, unbalanced]')
    my_parser.add_argument('--save_dir', type=str, default="/media/tohn/HDD2/Model_unlearn/Models_USAI/ViTModel", help='Main Output Path >> [ViT_unlearn: /media/HDD/mini-ImageNet/ViTModel_unlearn], [ViTunlearn_USAI: /media/HDD/mini-ImageNet/ViTModel_USAI]')
    my_parser.add_argument('--imgsize', type=int, default=384, help='Vit input size')
    my_parser.add_argument('--name', type=str, default=".", help='Name for Exp [exp1,...]')
    my_parser.add_argument('--R', type=int, help='[1:R1, 2:R2]')
    my_parser.add_argument('--lr', type=float, default=2e-5, help='[Train Unlearn: 1e-5, Train USAI: 2e-5]')
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
    if args.set == "ViT_unlearn": 
        _R = f"R{args.R}"
        if args.R == 1 and args.resume :
            print("**********[INFO]: Resume R1 VITl32 Unlearning model to Retrain**********")
            input_shape, model = loadresumemodel(args.checkpoint_dir)
        elif args.R == 2 and args.resume :
            print("**********[INFO]: Resume R2 VITl32 Unlearning model to Retrain**********")
            input_shape, model = loadresumemodel(args.checkpoint_dir)
        elif args.R == 1:
            print("**********[INFO]: Build VITl32 Unlearning model**********")
            input_shape, model = build_vit_unlearn(image_size = args.imgsize)
        elif args.R == 2:
            print("**********[INFO]: Load Model R1 to Train R2 Unlearning Model**********")
            input_shape, model = loadresumemodel(args.checkpoint_dir)
    #elif args.set == "ViTunlearn_USAI": 
    elif args.set == "ViTorigin_USAI": 
        _R = f'R{args.R}_{args.data}'
        if args.R == 1 and args.resume :
            print("**********[INFO]: Resume R1 VITl32 model to Retrain**********")
            input_shape, model = loadresumemodel_1input(args.checkpoint_dir)
        elif args.R == 2 and args.resume :
            print("**********[INFO]: Resume R2 VITl32 model to Retrain**********")
            input_shape, model = loadresumemodel_1input(args.checkpoint_dir)
        elif args.R == 1:
            print("**********[INFO]: Build VITl32 model**********")
            input_shape, model = build_ViTL32Model(IMAGE_SIZE = args.imgsize, Numclass=15)
        elif args.R == 2:
            print("**********[INFO]: Load Model R1 to Train R2 Model**********")
            input_shape, model = loadresumemodel_1input(args.checkpoint_dir)
        
    ## get images size 
    IMAGE_SIZE = input_shape[0]
    ### summary model
    model.summary()
    print('='*100)

    ## Create data training
    if args.set == "ViT_unlearn": 
        dataset = pd.read_csv(args.data_path, dtype=str)
        Train_df = dataset[dataset['subset']=='train'].reset_index(drop=True)        
        val_df = dataset[dataset['subset']=='val'].reset_index(drop=True)
        ### Implement > ## Train set  
        batch_generator_main, batch_generator_2 = batch_datagen(Train_df, input_shape, BATCH_SIZE=args.batchsize)
        batch_train = Flip_generator(batch_generator_main, batch_generator_2, IMAGE_SIZE=IMAGE_SIZE)
        ## Validation set 
        batchval_generator_main, batchval_generator_2 = batch_datagen(val_df, input_shape, BATCH_SIZE=args.batchsize)
        batch_val = Flip_generator(batchval_generator_main, batchval_generator_2, IMAGE_SIZE=IMAGE_SIZE)
    elif args.set == "ViTunlearn_USAI" or args.set == "ViTorigin_USAI": 
        if args.data == "balanced":
           ### Train set balanced classes 
           dataframe = pd.read_csv(f'{args.data_path}/Traindf_fold4_8_v1_Balance_classes.csv')
        elif args.data == "unbalanced":
            ### Train set imbalanced classes  
            dataframe = pd.read_csv(f'{args.data_path}/Traindf_fold4_8_v1.csv')
               
        ## validation set
        valframe = pd.read_csv(f'{args.data_path}/Valdf_fold3_v1.csv')
        ## Implement > ## Train set  
        train_generator, valid_generator = Data_generator(height=IMAGE_SIZE, width=IMAGE_SIZE, BATCH_SIZE=args.batchsize,
                                                                dataframe=dataframe, valframe=valframe)
    ## Create Main Output Path
    root_base = f'{args.save_dir}/{args.set}/{_R}/{args.name}'
    os.makedirs(root_base, exist_ok=True)
    ## Set mkdir TensorBoard 
    root_logdir = f"{root_base}/{args.tensorName}"
    os.makedirs(root_logdir, exist_ok=True)
    ### Run TensorBoard 
    run_logdir = get_run_logdir(root_logdir)
    #tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)
    tensorboard_cb = TensorBoard(log_dir=run_logdir, write_graph=False)
    ## Create Models Name 
    if args.set == "ViT_unlearn":
        modelName = f'modelVITL32_Unlearning_miniImageNet-{_R}.h5'
        on_epochName = f'modelVITL32_Unlearning_miniImageNet-{_R}'
    elif args.set == "ViTorigin_USAI":
        modelName = f'modelVITL32_USAI15ab-{_R}.h5'
        on_epochName = f'modelVITL32_USAI15ab-{_R}'
    elif args.set == "ViTunlearn_USAI" :
        modelName = f'modelVITL32_Unlearning_USAI15ab-{_R}.h5'
        on_epochName = f'modelVITL32_Unlearning_USAI15ab-{_R}'
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
    if args.set == "ViT_unlearn": 
        ### Set shows train steps:
        steps_per_epoch = cal_steps(len(Train_df), batch_size=args.batchsize)
        validation_steps = cal_steps(len(val_df), batch_size=args.batchsize)
        ### Set loss and opimizer 
        loss_smooth = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2)
        opt = tf.keras.optimizers.legacy.Adam(args.lr, decay=1e-4)
        #Compliermodel
        model.compile(optimizer=opt, loss=loss_smooth, metrics=['accuracy'])
        # Fit model 
        model.fit(batch_train, steps_per_epoch=steps_per_epoch, epochs=args.epochs, batch_size=args.batchsize,
                    validation_data=batch_val, validation_steps=validation_steps,
                    callbacks = [metrics_callback, tensorboard_cb])
    elif args.set == "ViTunlearn_USAI" or args.set == "ViTorigin_USAI" :
#          STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
#          STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
         ### Set loss and opimizer 
         losses = tf.keras.losses.categorical_crossentropy
         #opt = tf.keras.optimizers.legacy.Adam(args.lr, decay=2e-4)
         #opt = tf.keras.optimizers.RMSprop(learning_rate=args.lr)
         opt = RMSprop(learning_rate=args.lr)
         ### Compile model
         model.compile(loss=losses, optimizer=opt, metrics=['accuracy'])
         ## Fit model 
         model.fit(train_generator, epochs=args.epochs, batch_size=args.batchsize, validation_data=valid_generator,
                     callbacks = [metrics_callback, tensorboard_cb])
#          model.fit(x = avoid_error(train_generator), steps_per_epoch = STEP_SIZE_TRAIN,
#                               validation_data = valid_generator, validation_steps = STEP_SIZE_VALID,
#                               epochs = args.epochs, callbacks = [metrics_callback, tensorboard_cb])

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
    
    
## Run MAIN Function 
if __name__ == '__main__':
    main()
    
    

    
    
        
        






    







