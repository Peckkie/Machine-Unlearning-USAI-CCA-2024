import os
import tensorflow as tf
import glob
import shutil
import sys
import numpy as np
import pandas as pd
from EffNetmodels import loadresumemodel, loadmodelUnlearn, finetuneUSAI_B4, build_EffNetmodelB5, model_block5Unfreze, finetuneUSAI_B4ToB7, finetuneUSAI_B1ToB4, finetuneUSAI_B5ToB7
from ResNet152v2Model import build_baseResNet152v2, ResNetUnfreeze_conv3_block, ResNetUnfreeze_conv3to5_block, ResNetUnfreeze_conv1xto3_block, loadresumemodel_ResNet
from ResNet152v2Model import CreateResNet152v2modelUnlearn
from data_loader import Data_generator
from tensorflow.keras.models import load_model


def utils_createModel(network_name, sets, R, name, exp, resume=False, checkpoint_dir=None, Modeljson_dir=None, imgsize=None):
    ## Create Model
    if network_name == "EffNetB5":
        if sets == "MLunlearn_USAI": 
            if resume :
                input_shape, model = loadresumemodel(checkpoint_dir)
            elif R == 1 and name == "transfer" :
                print("[INFO]: Load ML Unlearn unfreezeB4 Model to Finetune Stage: Unfreeze FC Layers")
                input_shape, model = loadmodelUnlearn(checkpoint_dir)
            elif R == 2 and name == "unfreezeB4" :
                print("[INFO]: Load ML Unlearn unfreezeB4 Model to Finetune Stage: Unfreeze Block4")
                input_shape, model = finetuneUSAI_B4(Modeljson_dir, checkpoint_dir)
            elif R == 1 and name == "transfer" :
                print("[INFO]: Load ML Unlearn unfreezeB4-B7 Model to Finetune Stage: Unfreeze FC Layers")
                input_shape, model = loadmodelUnlearn(checkpoint_dir)
            elif R == 2 and name == "unfreezeB4-B7" :
                print("[INFO]: Load ML Unlearn unfreezeB4-B7 Model to Finetune Stage: Unfreeze Block4 to Block7")
                input_shape, model = finetuneUSAI_B4ToB7(Modeljson_dir, checkpoint_dir)
            elif R == 1 and name == "transfer" :
                print("[INFO]: Load ML Unlearn unfreezeB1-B1 Model to Finetune Stage: Unfreeze FC Layers")
                input_shape, model = loadmodelUnlearn(checkpoint_dir)
            elif R == 2 and name == "unfreezeB1-B4" :
                print("[INFO]: Load ML Unlearn unfreezeB1-B4 Model to Finetune Stage: Unfreeze Block1 to Block4")
                input_shape, model = finetuneUSAI_B1ToB4(Modeljson_dir, checkpoint_dir)
            elif R == 1 and name == "transfer" :
                print("[INFO]: Load ML Unlearn unfreeze Block5a_se_excite-Block7 Model to Finetune Stage: Unfreeze FC Layers")
                input_shape, model = loadmodelUnlearn(checkpoint_dir)
            elif R == 2 and name == "unfreezeBlock5a_se_excite" :
                print("[INFO]: Load ML Unlearn unfreezeB5-B7 Model to Finetune Stage: Unfreeze Block5a_se_excite to Block7")
                input_shape, model = finetuneUSAI_B5ToB7(Modeljson_dir, checkpoint_dir)
        elif sets == "MLorigin_USAI": 
            ## Create Model
            if resume :
                input_shape, model = loadresumemodel(checkpoint_dir)
            elif R == 1 and name == "transfer" :
                print("[INFO]: Build EffNetB5 Base Model to Transfer Learning Stage")
                input_shape, model = build_EffNetmodelB5(fine_tune=True, Numclasses=15)
            elif R == 2 and name == "Block5a_se_excite" :
                print("[INFO]: Load EffNetB5 Model to Finetune Stage: Unfreeze Block5a_se_excite Layer")
                input_shape, model = model_block5Unfreze(Modeljson_dir, checkpoint_dir)
    ## Create ResNet152V2 Model
    elif network_name == "ResNet152v2":
        if sets == "MLorigin_USAI": 
            if resume :
                (f"---------- [INFO]: Load ResNet152V2 Base Model to Resume Training R{R} Stage----------")
                input_shape, model = loadresumemodel_ResNet(Modeljson_dir, checkpoint_dir)
            elif R == 1 and name == "transfer" :
                print("---------- [INFO]: Build ResNet152V2 Base Model to Transfer Learning Stage ----------")
                input_shape, model = build_baseResNet152v2(imagesize=imgsize, Numclasses=15)
            elif R == 2 and name == "unfreeze_conv3_block" :
                print("---------- [INFO]: Load ResNet152V2 Base Model to Finetune Stage: unfreeze conv3_block ----------")
                input_shape, model = ResNetUnfreeze_conv3_block(Modeljson_dir, checkpoint_dir, sets)
            elif R == 2 and name == "unfreeze_conv3_block-conv5_block" :
                print("---------- [INFO]: Load ResNet152V2 Base Model to Finetune Stage: unfreeze conv3_block - conv5_block ----------")
                input_shape, model = ResNetUnfreeze_conv3to5_block(Modeljson_dir, checkpoint_dir, sets)
            elif R == 2 and name == "unfreeze_conv1-conv3_block" :
                print("---------- [INFO]: Load ResNet152V2 Base Model to Finetune Stage: unfreeze conv1x - conv3_block ----------")
                input_shape, model = ResNetUnfreeze_conv1xto3_block(Modeljson_dir, checkpoint_dir, sets)
        elif sets == "MLunlearn_USAI": 
             if resume :
                 (f"---------- [INFO]: Load ResNet152V2 Unlearn Model to Resume Training R{R} Stage, from {name} network ----------")
                 input_shape, model = loadresumemodel_ResNet(Modeljson_dir, checkpoint_dir)
             elif R == 1 and name == "transfer" :
                  print(f"---------- [INFO]: Create ResNet152V2 Unlearn Model to Transfer Learning Stage, from {exp} network ----------")
                  input_shape, model = CreateResNet152v2modelUnlearn(Modeljson_dir, checkpoint_dir, imgsize, Numclasses=15)
             elif R == 2 and name == "unfreeze_conv3_block" :
                  print("---------- [INFO]: Load ResNet152V2 Unlearn Model to Finetune Stage: unfreeze conv3_block ----------")
                  input_shape, model = ResNetUnfreeze_conv3_block(Modeljson_dir, checkpoint_dir, sets)
             elif R == 2 and name == "unfreeze_conv3_block-conv5_block" :
                  print("---------- [INFO]: Load ResNet152V2 Unlearn Model to Finetune Stage: unfreeze conv3_block - conv5_block ----------")
                  input_shape, model = ResNetUnfreeze_conv3to5_block(Modeljson_dir, checkpoint_dir, sets)
             elif R == 2 and name == "unfreeze_conv1-conv3_block" :
                  print("---------- [INFO]: Load ResNet152V2 Unlearn Model to Finetune Stage: unfreeze conv1x - conv3_block ----------")
                  input_shape, model = ResNetUnfreeze_conv1xto3_block(Modeljson_dir, checkpoint_dir, sets)
                
    return input_shape, model


    