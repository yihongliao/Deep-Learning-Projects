#!/usr/bin/env python

import random
import numpy
import torch
import os, sys


seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


from MyRegionProposalGenerator import *

rpg = RegionProposalGenerator(
                  dataroot_train = "/content/drive/MyDrive/HW6/data/Coco_train/",
                  dataroot_test  = "/content/drive/MyDrive/HW6/data/Coco_test/",
                  image_size = [128,128],
                  yolo_interval = 20,
                  path_saved_yolo_model = "/content/drive/MyDrive/HW6/yolo_model.pth",
                  momentum = 0.9,
                  learning_rate = 1e-5,
                  epochs = 20,
                  # epochs = 3,
                  batch_size = 1,
                  classes = ("airplane", "bus", "horse"),
                  use_gpu = True,
              )


yolo = RegionProposalGenerator.YoloLikeDetector( rpg = rpg )

# set the dataloaders
yolo.set_dataloaders(test=True)

model = yolo.NetForYolo(skip_connections=True, depth=8) 

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
num_layers = len(list(model.parameters()))
print("\n\nThe number of layers in the model: %d\n\n" % num_layers)

yolo.run_code_for_testing_multi_instance_detection(model, display_labels=True, display_images = True)

