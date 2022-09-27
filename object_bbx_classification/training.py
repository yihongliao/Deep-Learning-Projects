import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tvt
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import glob
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

from MyDLStudio import *


if __name__ == '__main__':

  dls = MyDLStudio(
            dataroot = "/content/drive/MyDrive/HW5/data/",
            image_size = [32,32],
            path_saved_model = "/content/drive/MyDrive/HW5/net.pth",
            momentum = 0.9,
            learning_rate = 1e-4,
            epochs = 2,
            batch_size = 4,
            classes = ["cat", "train", "elephant", "airplane", "giraffe"],
            use_gpu = True,
          )

  detector = MyDLStudio.DetectAndLocalize( dl_studio = dls ) 

  dataserver_train = MyDLStudio.DetectAndLocalize.cocoDataset(
                                  train_or_test = 'train',
                                  coco_json_path = '/content/drive/MyDrive/HW5/',
                                  dl_studio = dls, 
                                  images_per_class = 1000)

  detector.dataserver_train = dataserver_train

  detector.load_coco_training_dataset(dataserver_train)

  model = detector.MyLOADnet(skip_connections=True, depth=8) #temporary

  detector.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model)
