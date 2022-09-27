import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tvt
import torchvision.transforms.functional as F
import torchvision.utils as tutils
import torch.optim as optim
import numpy as np
import time

from PIL import Image
from PIL import ImageDraw
from PIL import ImageTk
from PIL import ImageFont
import sys,os,os.path,glob,signal
import re
import functools
import math
import random
import copy
import gzip
import pickle
if sys.version_info[0] == 3:
    import tkinter as Tkinter
    from tkinter.constants import *
else:
    import Tkinter    
    from Tkconstants import *

import matplotlib.pyplot as plt
import logging                        ## for suppressing matplotlib warning messages
from pycocotools.coco import COCO
import requests
import json
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

#______________________________  RegionProposalGenerator Class Definition  ________________________________

class RegionProposalGenerator(object):

    # Class variables: 
    region_mark_coords = {}
    drawEnable = startX = startY = 0
    canvas = None

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''RegionProposalGenerator constructor can only be called with keyword arguments for 
                      the following keywords: dataroot_train, dataroot_test, image_size, data_image, 
                      binary_or_gray_or_color, kay, image_size_reduction_factor, max_iterations, sigma, 
                      image_normalization_required, momentum, min_size_for_graph_based_blobs, 
                      max_num_blobs_expected, path_saved_RPN_model, path_saved_single_instance_detector_model,
                      path_saved_yolo_model, learning_rate, epochs, batch_size, classes, debug_train, 
                      debug_test, use_gpu, color_homogeneity_thresh, gray_var_thresh, texture_homogeneity_thresh, 
                      yolo_interval, and debug''')
        dataroot_train = dataroot_test = data_image = sigma = image_size_reduction_factor = kay = momentum = None
        learning_rate = epochs = min_size_for_graph_based_blobs = max_num_blobs_expected = path_saved_RPN_model = None
        path_saved_single_instance_detector_model = batch_size = use_gpu = binary_or_gray_or_color =  max_iterations = None
        image_normalization_required = classes = debug_train = color_homogeneity_thresh = gray_var_thresh = None
        image_size = texture_homogeneity_thresh = debug = debug_test = path_saved_yolo_model = yolo_interval = None

        if 'dataroot_train' in kwargs                :   dataroot_train = kwargs.pop('dataroot_train')
        if 'dataroot_test' in kwargs                 :   dataroot_test = kwargs.pop('dataroot_test')
        if 'image_size' in kwargs                    :   image_size = kwargs.pop('image_size')
        if 'path_saved_RPN_model' in kwargs          :   path_saved_RPN_model = kwargs.pop('path_saved_RPN_model')
        if 'path_saved_single_instance_detector_model' in kwargs   :   
                 path_saved_single_instance_detector_model = kwargs.pop('path_saved_single_instance_detector_model')
        if 'path_saved_yolo_model' in kwargs         :   path_saved_yolo_model = kwargs.pop('path_saved_yolo_model')
        if 'yolo_interval' in kwargs                 :   yolo_interval = kwargs.pop('yolo_interval')

        if 'momentum' in kwargs                      :   momentum = kwargs.pop('momentum')
        if 'learning_rate' in kwargs                 :   learning_rate = kwargs.pop('learning_rate')
        if 'epochs' in kwargs                        :   epochs = kwargs.pop('epochs')
        if 'batch_size' in kwargs                    :   batch_size = kwargs.pop('batch_size')
        if 'classes' in kwargs                       :   classes = kwargs.pop('classes')
        if 'debug_train' in kwargs                   :   debug_train = kwargs.pop('debug_train')
        if 'debug_test' in kwargs                    :   debug_test = kwargs.pop('debug_test')
        if 'use_gpu' in kwargs                       :   use_gpu = kwargs.pop('use_gpu')
        if 'data_image' in kwargs                    :   data_image = kwargs.pop('data_image')
        if 'sigma' in kwargs                         :   sigma = kwargs.pop('sigma')
        if 'kay' in kwargs                           :   kay = kwargs.pop('kay')
        if 'image_size_reduction_factor' in kwargs   :   image_size_reduction_factor = kwargs.pop('image_size_reduction_factor')
        if 'binary_or_gray_or_color' in kwargs       :   binary_or_gray_or_color = kwargs.pop('binary_or_gray_or_color')
        if 'image_normalization_required' in kwargs  :   image_normalization_required = kwargs.pop('image_normalization_required')
        if 'max_iterations' in kwargs                :   max_iterations=kwargs.pop('max_iterations')
        if 'color_homogeneity_thresh' in kwargs      :   color_homogeneity_thresh = kwargs.pop('color_homogeneity_thresh')
        if 'gray_var_thresh' in kwargs               :    gray_var_thresh = kwargs.pop('gray_var_thresh')
        if 'texture_homogeneity_thresh' in kwargs    :   texture_homogeneity_thresh = kwargs.pop('texture_homogeneity_thresh')
        if 'min_size_for_graph_based_blobs' in kwargs :  min_size_for_graph_based_blobs = kwargs.pop('min_size_for_graph_based_blobs')
        if 'max_num_blobs_expected' in kwargs        :  max_num_blobs_expected = kwargs.pop('max_num_blobs_expected')
        if 'debug' in kwargs                         :   debug = kwargs.pop('debug') 
#        if len(kwargs) != 0: raise ValueError('''You have provided unrecognizable keyword args''')
        if dataroot_train:
            self.dataroot_train = dataroot_train
        if dataroot_test:
            self.dataroot_test = dataroot_test
        if image_size:   
            self.image_size = image_size      
        if  path_saved_RPN_model:
            self.path_saved_RPN_model = path_saved_RPN_model
        if  path_saved_single_instance_detector_model:
            self.path_saved_single_instance_detector_model = path_saved_single_instance_detector_model
        if  path_saved_yolo_model:
            self.path_saved_yolo_model = path_saved_yolo_model
        if yolo_interval:
            self.yolo_interval = yolo_interval
        if classes:
            self.class_labels = classes
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 1e-6
        if momentum:
            self.momentum = momentum
        if epochs:
            self.epochs = epochs
        if batch_size:
            self.batch_size = batch_size
        if use_gpu is not None:
            self.use_gpu = use_gpu
            if use_gpu is True:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda:0")
                else:
                    raise Exception("You requested GPU support, but there's no GPU on this machine")
            else:
                self.device = torch.device("cpu")
        if debug_train:                             
            self.debug_train = debug_train
        else:
            self.debug_train = 0
        if debug_test:                             
            self.debug_test = debug_test
        else:
            self.debug_test = 0
        if data_image: 
            self.data_im_name = data_image
            self.data_im =  Image.open(data_image)
            self.original_im = Image.open(data_image)
        if binary_or_gray_or_color:
            self.binary_or_gray_or_color = binary_or_gray_or_color
        if sigma is not None: 
            self.sigma = sigma
        else:
            self.sigma = 0
        if kay is not None:   self.kay = kay
        if image_size_reduction_factor is not None:
            self.image_size_reduction_factor = image_size_reduction_factor
        else:
            self.image_size_reduction_factor = 1
        if image_normalization_required is not None:
            self.image_normalization_required = image_normalization_required
        else:
            self.image_normalization_required = False
        if max_iterations is not None:
            self.max_iterations = max_iterations
        else:
            self.max_iterations = 40
        if color_homogeneity_thresh is not None:
            self.color_homogeneity_thresh = color_homogeneity_thresh
        if gray_var_thresh is not None:
            self.gray_var_thresh = gray_var_thresh
        if texture_homogeneity_thresh is not None:
            self.texture_homogeneity_thresh = texture_homogeneity_thresh
        if min_size_for_graph_based_blobs is not None:
            self.min_size_for_graph_based_blobs = min_size_for_graph_based_blobs
        if max_num_blobs_expected is not None:
            self.max_num_blobs_expected = max_num_blobs_expected
        self.image_portion_delineation_coords = []
        if debug:                             
            self.debug = debug
        else:
            self.debug = 0
        self.iterations_used = 0

    class cocoDataset(torch.utils.data.Dataset):
      def __init__(self, rpg, train_or_test, dataroot_train=None, dataroot_test=None, transform=None):
        super(RegionProposalGenerator.cocoDataset, self).__init__()
        self.rpg = rpg
        self.train_or_test = train_or_test
        self.dataroot_train = dataroot_train
        self.dataroot_test  = dataroot_test
        self.database_train = {}
        self.database_test = {}
        self.dataset_size_train = 0
        self.dataset_size_test = 0
        self.class_list = self.rpg.class_labels
        self.coco_json_path = "/content/drive/MyDrive/HW6/";
              

        if train_or_test == 'train':         
          if os.path.isfile(self.dataroot_train + "torch_saved_Coco_multi_dataset_train_1500.pt"):
            print("\nLoading training data from torch saved file")
            self.database_train = torch.load(self.dataroot_train + "torch_saved_Coco_multi_dataset_train_1500.pt")
            self.dataset_size_train =  len(self.database_train)          
          else:
            # initialize COCO api for instance annotations
            self.coco = COCO(self.coco_json_path + 'instances_train2017.json')
            self.class_ids = self.coco.getCatIds(catNms=self.class_list)
            # obj_class_ids = sorted(self.class_ids)
            obj_class_ids = self.class_ids
            self.obj_class_id_dict = {obj_class_ids[i] : i for i in range(len(obj_class_ids))}
            print(self.obj_class_id_dict)

            print('Downloading training data images')
            self.download_images(train_or_test)

            print('Saving training data as torch saved file')
            torch.save(self.database_train, self.dataroot_train + "torch_saved_Coco_multi_dataset_train_1500.pt")

          print("Number of training data: ", self.dataset_size_train)

        else:    
          if os.path.isfile(self.dataroot_test + "torch_saved_Coco_multi_dataset_test_1000.pt"):
            print("\nLoading testing data from torch saved file")
            self.database_test = torch.load(self.dataroot_test + "torch_saved_Coco_multi_dataset_test_1000.pt")
            self.dataset_size_test =  len(self.database_test)                    
          else:
            # initialize COCO api for instance annotations
            self.coco = COCO(self.coco_json_path + 'instances_val2017.json')
            self.class_ids = self.coco.getCatIds(catNms=self.class_list)
            # obj_class_ids = sorted(self.class_ids)
            obj_class_ids = self.class_ids
            self.obj_class_id_dict = {obj_class_ids[i] : i for i in range(len(obj_class_ids))}
            print(self.obj_class_id_dict)

            print('Downloading testing data images')
            self.download_images(train_or_test)

            print('Saving testing data as torch saved file')
            torch.save(self.database_test, self.dataroot_test + "torch_saved_Coco_multi_dataset_test_1000.pt")

          print("Number of testing data: ", self.dataset_size_test)

      def download_images(self, train_or_test):
        print("class IDs: ", self.class_ids)

        max_num_objects_in_image = 5
        num_valid_image = 0
        num_valid_image_per_class = 0

        if train_or_test == 'train':
          image_folder = self.dataroot_train + "images/"
          total_num_valid_image_per_class = 500
        else:
          image_folder = self.dataroot_test + "images/"
          total_num_valid_image_per_class = 20

        # create images folder
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        for category in self.class_list:

          catId = self.coco.getCatIds(catNms=category);
          imgIds = self.coco.getImgIds(catIds=catId)
          print('Image pool: ', len(imgIds))
          img_dicts = self.coco.loadImgs(imgIds)

          num_valid_image_per_class = 0
          i = 0
          while (num_valid_image_per_class < total_num_valid_image_per_class):
            annId = self.coco.getAnnIds(imgIds=imgIds[i], iscrowd=False)
            anns = self.coco.loadAnns(annId)
            img_dict = img_dicts[i]
            w0 = img_dict['width']
            h0 = img_dict['height']
            w_scale = (self.rpg.image_size[0]-1)/(w0-1)
            h_scale = (self.rpg.image_size[1]-1)/(h0-1)
            
            num_objects_in_image = 0
            for ann in anns:
              if ann['bbox'][2] > 4.0/w_scale and ann['bbox'][3] > 4.0/w_scale:
                for class_id in self.class_ids:               
                  if ann['category_id'] == class_id:
                    num_objects_in_image += 1

            if num_objects_in_image >= 2: 

              if num_valid_image_per_class%50 == 49:
                print(num_valid_image_per_class, "/", total_num_valid_image_per_class)

              # download image             
              img_url = img_dict['coco_url']
              if len(img_url) <= 1:
                print("url error")
                return False
              try: 
                img_response = requests.get(img_url, timeout = 10)
              except Exception as e:
                print("request error")
                return False

              img_name = img_url.split('/')[-1]

              img_path = os.path.join(image_folder, img_name)
              with open(img_path, 'wb') as file:
                file.write(img_response.content)
              im = Image.open(img_path)
              if im.mode != "RGB":
                im = im.convert(mode="RGB")
              

              # resize image
              im_resized = im.resize(self.rpg.image_size, Image.BOX)
              np_im_resized = np.array(im_resized)

              im_resized.save(img_path)
              
              # create image annotation
              annotation = {}
              bboxes = {}
              bbox_labels = {}
              j = 0; 
              for ann in anns:
                if j < max_num_objects_in_image:
                  if ann['bbox'][2] > 4.0/w_scale and ann['bbox'][3] > 4.0/w_scale:
                    for class_id in self.class_ids:
                      if ann['category_id'] == class_id:
                        bbox_x0 = (ann['bbox'][0] + 1 - 1)*w_scale + 1 - 1
                        bbox_x1 = ((ann['bbox'][0] + 1 + ann['bbox'][2] - 1) - 1)*w_scale + 1 - 1
                        bbox_y0 = (ann['bbox'][1] + 1 - 1)*h_scale + 1 - 1
                        bbox_y1 = ((ann['bbox'][1] + 1 + ann['bbox'][3] - 1) - 1)*h_scale + 1 - 1
                        bboxes[j] = [bbox_x0, bbox_y0, bbox_x1, bbox_y1]
                        bbox_labels[j] = self.obj_class_id_dict[ann['category_id']] 
                        j += 1

              annotation['filname'] = img_name
              annotation['num_objects'] = j
              annotation['bboxes'] = bboxes
              annotation['bbox_labels'] = bbox_labels

              if train_or_test == 'train':
                self.database_train[num_valid_image] = [img_path, im_resized, annotation]     
              else:
                self.database_test[num_valid_image] = [img_path, im_resized, annotation]   

              num_valid_image += 1
              num_valid_image_per_class += 1
            i += 1

        if train_or_test == 'train':
          self.dataset_size_train = num_valid_image
        else:
          self.dataset_size_test = num_valid_image
              
      def __len__(self):
        if self.train_or_test == 'train':
            return self.dataset_size_train
        elif self.train_or_test == 'test':
            return self.dataset_size_test

      def __getitem__(self, idx):
        if self.train_or_test == 'train':       
            image_path, image, annotation = self.database_train[idx]
        elif self.train_or_test == 'test':
            image_path, image, annotation = self.database_test[idx]

        im_tensor = tvt.ToTensor()(image)
        bbox_tensor     = torch.zeros(5,4, dtype=torch.uint8)
        bbox_label_tensor    = torch.zeros(5, dtype=torch.uint8) + 13
        num_objects_in_image = annotation['num_objects']
        
        for i in range(num_objects_in_image):
            bbox     = annotation['bboxes'][i]
            label    = annotation['bbox_labels'][i]
            bbox_label_tensor[i] = label
            bbox_tensor[i] = torch.LongTensor(bbox)      
        return im_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image

    class YoloLikeDetector(nn.Module): 
      def __init__(self, rpg):
            super(RegionProposalGenerator.YoloLikeDetector, self).__init__()
            self.rpg = rpg
            self.train_dataloader = None
            self.test_dataloader = None
      def set_dataloaders(self, train=False, test=False):
            if train:
                dataserver_train = RegionProposalGenerator.cocoDataset(self.rpg, 
                                                       "train", dataroot_train=self.rpg.dataroot_train)
                self.train_dataloader = torch.utils.data.DataLoader(dataserver_train, 
                                                      self.rpg.batch_size, shuffle=True, num_workers=2)
            if test:
                dataserver_test = RegionProposalGenerator.cocoDataset(self.rpg, 
                                                          "test", dataroot_test=self.rpg.dataroot_test)
                self.test_dataloader = torch.utils.data.DataLoader(dataserver_test, 
                                                     self.rpg.batch_size, shuffle=False, num_workers=2)

      class SkipBlock(nn.Module):
          """
          Class Path:   RegionProposalGenerator  ->  YoloLikeDetector  ->  SkipBlock
          """
          def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
            super(RegionProposalGenerator.YoloLikeDetector.SkipBlock, self).__init__()
            self.downsample = downsample
            self.skip_connections = skip_connections
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            self.convo3 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            norm_layer1 = nn.BatchNorm2d
            norm_layer2 = nn.BatchNorm2d
            norm_layer3 = nn.BatchNorm2d
            self.bn1 = norm_layer1(out_ch)
            self.bn2 = norm_layer2(out_ch)
            self.bn3 = norm_layer3(out_ch)
            if downsample:
                self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
          def forward(self, x):
            identity = x                                     
            out = self.convo1(x)                              
            out = self.bn1(out)                              
            out = torch.nn.functional.relu(out)
            out1 = out.clone()
            if self.in_ch == self.out_ch:
                out = self.convo2(out)                              
                out = self.bn2(out)                              
                out = torch.nn.functional.relu(out)
                out = self.convo3(out)                              
                out = self.bn3(out)                              
                out = torch.nn.functional.relu(out)               
            if self.skip_connections:
                if self.in_ch == self.out_ch:
                    out += out1 + identity                              
                else:
                    out[:,:self.in_ch,:,:] += out1[:,:self.in_ch,:,:] + identity
                    out[:,self.in_ch:,:,:] += out1[:,self.in_ch:,:,:] + identity
            if self.downsample:
                out = self.downsampler(out)
                identity = self.downsampler(identity)

            return out

      class NetForYolo(nn.Module):
            def __init__(self, skip_connections=True, depth=8):
                super(RegionProposalGenerator.YoloLikeDetector.NetForYolo, self).__init__()
                if depth not in [8,10,12,14,16]:
                    sys.exit("This network has only been tested for 'depth' values 8, 10, 12, 14, and 16")
                self.depth = depth // 2
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.bn1  = nn.BatchNorm2d(64)
                self.bn2  = nn.BatchNorm2d(128)
                self.bn3  = nn.BatchNorm2d(256)
                self.skip64_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64_arr.append(RegionProposalGenerator.YoloLikeDetector.SkipBlock(64, 64,
                                                                                    skip_connections=skip_connections))
                self.skip64ds = RegionProposalGenerator.YoloLikeDetector.SkipBlock(64,64,downsample=True, 
                                                                                     skip_connections=skip_connections)
                self.skip64to128 = RegionProposalGenerator.YoloLikeDetector.SkipBlock(64, 128, 
                                                                                    skip_connections=skip_connections )
                self.skip128_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128_arr.append(RegionProposalGenerator.YoloLikeDetector.SkipBlock(128,128,
                                                                                    skip_connections=skip_connections))
                self.skip128ds = RegionProposalGenerator.YoloLikeDetector.SkipBlock(128,128,
                                                                    downsample=True, skip_connections=skip_connections)
                self.skip128to256 = RegionProposalGenerator.YoloLikeDetector.SkipBlock(128, 256, 
                                                                                    skip_connections=skip_connections )
                self.skip256_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip256_arr.append(RegionProposalGenerator.YoloLikeDetector.SkipBlock(256,256,
                                                                                    skip_connections=skip_connections))
                self.skip256ds = RegionProposalGenerator.YoloLikeDetector.SkipBlock(256,256,
                                                                    downsample=True, skip_connections=skip_connections)
                self.fc_seqn = nn.Sequential(
                    nn.Linear(8192, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, 2048),
                    nn.ReLU(inplace=True),
#                    nn.Linear(2048, 1440)
                    nn.Linear(2048, 1620)
                )

            def forward(self, x):
                x = self.pool(torch.nn.functional.relu(self.conv1(x)))          
                x = nn.MaxPool2d(2,2)(torch.nn.functional.relu(self.conv2(x)))       
                for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
                    x = skip64(x)                
                x = self.skip64ds(x)
                for i,skip64 in enumerate(self.skip64_arr[self.depth//4:]):
                    x = skip64(x)                
                x = self.bn1(x)
                x = self.skip64to128(x)
                for i,skip128 in enumerate(self.skip128_arr[:self.depth//4]):
                    x = skip128(x)                
                x = self.bn2(x)
                x = self.skip128ds(x)
                x = x.view(-1, 8192 )
                x = self.fc_seqn(x)
                return x

      def run_code_for_training_multi_instance_detection(self, net, display_labels=False, display_images=False):        
          if self.rpg.batch_size > 1:                                                                                    ## (1)
              sys.exit("YOLO-like multi-instance object detection has only been tested for batch_size of 1")             ## (2)
          yolo_debug = False
          filename_for_out1 = "performance_numbers_" + str(self.rpg.epochs) + "label.txt"                                
          filename_for_out2 = "performance_numbers_" + str(self.rpg.epochs) + "regres.txt"                               
          FILE1 = open(filename_for_out1, 'w')                                                                           
          FILE2 = open(filename_for_out2, 'w')                                                                           
          net = net.to(self.rpg.device)                                                                                  
          criterion1 = nn.BCELoss()                    # For the first element of the 8 element yolo vector              ## (3)
          criterion2 = nn.MSELoss()                    # For the regiression elements (indexed 2,3,4,5) of yolo vector   ## (4)
          criterion3 = nn.CrossEntropyLoss()           # For the last three elements of the 8 element yolo vector        ## (5)
          print("\n\nLearning Rate: ", self.rpg.learning_rate)
          optimizer = optim.SGD(net.parameters(), lr=self.rpg.learning_rate, momentum=self.rpg.momentum)                 ## (6)
          print("\n\nStarting training loop...\n\n")
          start_time = time.perf_counter()
          Loss_tally = []
          elapsed_time = 0.0
          yolo_interval = self.rpg.yolo_interval                                                                         ## (7)
          num_yolo_cells = (self.rpg.image_size[0] // yolo_interval) * (self.rpg.image_size[1] // yolo_interval)         ## (8)
          num_anchor_boxes =  5    # (height/width)   1/5  1/3  1/1  3/1  5/1                                            ## (9)
          max_obj_num  = 5                                                                                               ## (10)
          ## The 8 in the following is the size of the yolo_vector for each anchor-box in a given cell.  The 8 elements 
          ## are: [obj_present, bx, by, bh, bw, c1, c2, c3] where bx and by are the delta diffs between the centers
          ## of the yolo cell and the center of the object bounding box in terms of a unit for the cell width and cell 
          ## height.  bh and bw are the height and the width of object bounding box in terms of the cell height and width.
          for epoch in range(self.rpg.epochs):                                                                           ## (11)
              print("epoch: ", epoch)
              running_loss = 0.0                                                                                         ## (12)
              for iter, data in enumerate(self.train_dataloader):   
                  if yolo_debug:
                      print("\n\n\n======================================= iteration: %d ========================================\n" % iter)
                  yolo_tensor = torch.zeros( self.rpg.batch_size, num_yolo_cells, num_anchor_boxes, 8 )                  ## (13)
                  # im_tensor, seg_mask_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data                ## (14)
                  im_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data                ## (14)
                  im_tensor   = im_tensor.to(self.rpg.device)                                                            ## (15)
                  # seg_mask_tensor = seg_mask_tensor.to(self.rpg.device)                 
                  bbox_tensor = bbox_tensor.to(self.rpg.device)
                  bbox_label_tensor = bbox_label_tensor.to(self.rpg.device)
                  yolo_tensor = yolo_tensor.to(self.rpg.device)
                  if yolo_debug:
                      logger = logging.getLogger()
                      old_level = logger.level
                      logger.setLevel(100)
                      plt.figure(figsize=[15,4])
                      plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor,normalize=True,padding=3,pad_value=255).cpu(), (1,2,0)))
                      plt.savefig("test.png")
                      plt.show()
                      logger.setLevel(old_level)
                  cell_height = yolo_interval                                                                            ## (16)
                  cell_width = yolo_interval                                                                             ## (17)
                  if yolo_debug:
                      print("\n\nnum_objects_in_image: ")
                      print(num_objects_in_image)
                  num_cells_image_width = self.rpg.image_size[0] // yolo_interval                                        ## (18)
                  num_cells_image_height = self.rpg.image_size[1] // yolo_interval                                       ## (19)
                  height_center_bb = torch.zeros(im_tensor.shape[0], 1).float().to(self.rpg.device)                      ## (20)
                  width_center_bb = torch.zeros(im_tensor.shape[0], 1).float().to(self.rpg.device)                       ## (21)
                  obj_bb_height = torch.zeros(im_tensor.shape[0], 1).float().to(self.rpg.device)                         ## (22)
                  obj_bb_width =  torch.zeros(im_tensor.shape[0], 1).float().to(self.rpg.device)                         ## (23)

                  ## idx is for object index
                  for idx in range(max_obj_num):                                                                         ## (24)
                      ## In the mask, 1 means good image instance in batch, 0 means bad image instance in batch
#                        batch_mask = torch.ones( self.rpg.batch_size, dtype=torch.int8).to(self.rpg.device)
                      if yolo_debug:
                          print("\n\n               ================  object indexed %d ===============              \n\n" % idx)
                      ## Note that the bounding-box coordinates are in the (x,y) format, with x-positive going to
                      ## right and the y-positive going down. A bbox is specified by (x_min,y_min,x_max,y_max):
                      if yolo_debug:
                          print("\n\nshape of bbox_tensor: ", bbox_tensor.shape)
                          print("\n\nbbox_tensor:")
                          print(bbox_tensor)
                      ## in what follows, the first index (set to 0) is for the batch axis
                      height_center_bb =  (bbox_tensor[0,idx,1] + bbox_tensor[0,idx,3]) // 2                             ## (25)
                      width_center_bb =  (bbox_tensor[0,idx,0] + bbox_tensor[0,idx,2]) // 2                              ## (26)
                      obj_bb_height = bbox_tensor[0,idx,3] -  bbox_tensor[0,idx,1]                                       ## (27)
                      obj_bb_width = bbox_tensor[0,idx,2] - bbox_tensor[0,idx,0]                                         ## (28)
                      
                      if (obj_bb_height < 4.0) or (obj_bb_width < 4.0): continue                                         ## (29)
                      
                      cell_row_indx =  (height_center_bb / yolo_interval).int()          ## for the i coordinate         ## (30)
                      cell_col_indx =  (width_center_bb / yolo_interval).int()           ## for the j coordinates        ## (31)
                      cell_row_indx = torch.clamp(cell_row_indx, max=num_cells_image_height - 1)                         ## (32)
                      cell_col_indx = torch.clamp(cell_col_indx, max=num_cells_image_width - 1)                          ## (33)

                      ## The bh and bw elements in the yolo vector for this object:  bh and bw are measured relative 
                      ## to the size of the grid cell to which the object is assigned.  For example, bh is the 
                      ## height of the bounding-box divided by the actual height of the grid cell.
                      bh  =  obj_bb_height.float() / yolo_interval                                                       ## (34)
                      bw  =  obj_bb_width.float()  / yolo_interval                                                       ## (35)

                      ## You have to be CAREFUL about object center calculation since bounding-box coordinates
                      ## are in (x,y) format --- with x-positive going to the right and y-positive going down.
                      obj_center_x =  (bbox_tensor[0,idx][2].float() +  bbox_tensor[0,idx][0].float()) / 2.0             ## (36)
                      obj_center_y =  (bbox_tensor[0,idx][3].float() +  bbox_tensor[0,idx][1].float()) / 2.0             ## (37)
                      ## Now you need to switch back from (x,y) format to (i,j) format:
                      yolocell_center_i =  cell_row_indx*yolo_interval + float(yolo_interval) / 2.0                      ## (38)
                      yolocell_center_j =  cell_col_indx*yolo_interval + float(yolo_interval) / 2.0                      ## (39)
                      del_x  =  (obj_center_x.float() - yolocell_center_j.float()) / yolo_interval                       ## (40)
                      del_y  =  (obj_center_y.float() - yolocell_center_i.float()) / yolo_interval                       ## (41)
                      class_label_of_object = bbox_label_tensor[0,idx].item()                                            ## (42)
                      ## When batch_size is only 1, it is easy to discard an image that has no known objects in it.
                      ## To generalize this notion to arbitrary batch sizes, you will need a batch mask to indicate
                      ## the images in a batch that should not be considered in the rest of this code.                     
                      if class_label_of_object == 13: continue                                                           ## (43)
                      AR = obj_bb_height.float() / obj_bb_width.float()                                                  ## (44)
                      if AR <= 0.2:               anch_box_index = 0                                                     ## (45)
                      if 0.2 < AR <= 0.5:         anch_box_index = 1                                                     ## (46)
                      if 0.5 < AR <= 1.5:         anch_box_index = 2                                                     ## (47)
                      if 1.5 < AR <= 4.0:         anch_box_index = 3                                                     ## (48)
                      if AR > 4.0:                anch_box_index = 4                                                     ## (49)
                      yolo_vector = torch.FloatTensor([0,del_x.item(), del_y.item(), bh.item(), bw.item(), 0, 0, 0] )    ## (50)
                      yolo_vector[0] = 1                                                                                 ## (51)
                      yolo_vector[5 + class_label_of_object] = 1                                                         ## (52)
                      yolo_cell_index =  cell_row_indx.item() * num_cells_image_width  +  cell_col_indx.item()           ## (53)
                      yolo_tensor[0,yolo_cell_index, anch_box_index] = yolo_vector                                       ## (54)
                      yolo_tensor_aug = torch.zeros(self.rpg.batch_size, num_yolo_cells, \
                                                                  num_anchor_boxes,9).float().to(self.rpg.device)         ## (55) 
                      yolo_tensor_aug[:,:,:,:-1] =  yolo_tensor                                                          ## (56)
                      if yolo_debug: 
                          print("\n\nyolo_tensor specific: ")
                          print(yolo_tensor[0,18,2])
                          print("\nyolo_tensor_aug_aug: ") 
                          print(yolo_tensor_aug[0,18,2])
                          print(yolo_tensor[0, yolo_cell_index, anch_box_index])
                  ## If no object is present, throw all the prob mass into the extra 9th ele of yolo_vector
                  for icx in range(num_yolo_cells):                                                                      ## (57)
                      for iax in range(num_anchor_boxes):                                                                ## (58)
                          if yolo_tensor_aug[0,icx,iax,0] == 0:                                                          ## (59)
                              yolo_tensor_aug[0,icx,iax,-1] = 1                                                          ## (60)
                  if yolo_debug:
                      logger = logging.getLogger()
                      old_level = logger.level
                      logger.setLevel(100)
                      plt.figure(figsize=[15,4])
                      plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                        padding=3, pad_value=255).cpu(), (1,2,0)))
                      plt.show()

                  optimizer.zero_grad()                                                                                  ## (61)
                  output = net(im_tensor)                                                                                ## (62)
                  predictions_aug = output.view(self.rpg.batch_size,num_yolo_cells,num_anchor_boxes,9)                   ## (63)
                  loss = torch.tensor(0.0, requires_grad=True).float().to(self.rpg.device)                               ## (64)
                  for icx in range(num_yolo_cells):                                                                      ## (65)
                      for iax in range(num_anchor_boxes):                                                                ## (66)
                          pred_yolo_vector = predictions_aug[0,icx,iax]                                                  ## (67)
                          target_yolo_vector = yolo_tensor_aug[0,icx,iax]                                                ## (68)
                          ##  Estiming presence/absence of object and the Binary Cross Entropy section:
                          object_presence = nn.Sigmoid()(torch.unsqueeze(pred_yolo_vector[0], dim=0))                    ## (69)
                          target_for_prediction = torch.unsqueeze(target_yolo_vector[0], dim=0)                          ## (70)
                          bceloss = criterion1(object_presence, target_for_prediction)                                   ## (71)
                          loss += bceloss                                                                                ## (72)
                          ## MSE section for regression params:
                          pred_regression_vec = pred_yolo_vector[1:5]                                                    ## (73)
                          pred_regression_vec = torch.unsqueeze(pred_regression_vec, dim=0)                              ## (74)
                          target_regression_vec = torch.unsqueeze(target_yolo_vector[1:5], dim=0)                        ## (75)
                          regression_loss = criterion2(pred_regression_vec, target_regression_vec)                       ## (76)
                          loss += regression_loss                                                                        ## (77)
                          ##  CrossEntropy section for object class label:
                          probs_vector = pred_yolo_vector[5:]                                                            ## (78)
                          probs_vector = torch.unsqueeze( probs_vector, dim=0 )                                          ## (79)
                          target = torch.argmax(target_yolo_vector[5:])                                                  ## (80)
                          target = torch.unsqueeze( target, dim=0 )                                                      ## (81)
                          class_labeling_loss = criterion3(probs_vector, target)                                         ## (82)
                          loss += class_labeling_loss                                                                    ## (83)
                  if yolo_debug:
                      print("\n\nshape of loss: ", loss.shape)
                      print("\n\nloss: ", loss)
                  loss.backward()                                                                                        ## (84)
                  optimizer.step()                                                                                       ## (85)
                  running_loss += loss.item()                                                                            ## (86)
                  if iter%100==99:                                                                                     ## (87)
                      if display_images:
                          print("\n\n\n")                ## for vertical spacing for the image to be displayed later
                      current_time = time.perf_counter()
                      elapsed_time = current_time - start_time 
                      avg_loss = running_loss / float(100)                                                              ## (88)
                      print("\n[epoch:%d/%d, iter=%4d  elapsed_time=%5d secs]      mean value for loss: %7.4f" % 
                                                          (epoch+1,self.rpg.epochs, iter+1, elapsed_time, avg_loss))     ## (89)
                      Loss_tally.append(running_loss)
                      FILE1.write("%.3f\n" % avg_loss)
                      FILE1.flush()
                      running_loss = 0.0                                                                                 ## (90)
                      if display_labels:
                          predictions = output.view(self.rpg.batch_size,num_yolo_cells,num_anchor_boxes,9)               ## (91)
                          if yolo_debug:
                              print("\n\nyolo_vector for first image in batch, cell indexed 18, and AB indexed 2: ")
                              print(predictions[0, 18, 2])
                          for ibx in range(predictions.shape[0]):                             # for each batch image     ## (92)
                              icx_2_best_anchor_box = {ic : None for ic in range(36)}                                    ## (93)
                              for icx in range(predictions.shape[1]):                         # for each yolo cell       ## (94)
                                  cell_predi = predictions[ibx,icx]                                                      ## (95)
                                  prev_best = 0                                                                          ## (96)
                                  for anchor_bdx in range(cell_predi.shape[0]):                                          ## (97)
                                      if cell_predi[anchor_bdx][0] > cell_predi[prev_best][0]:                           ## (98)
                                          prev_best = anchor_bdx                                                         ## (99)
                                  best_anchor_box_icx = prev_best                                                        ## (100)
                                  icx_2_best_anchor_box[icx] = best_anchor_box_icx                                       ## (101)
                              sorted_icx_to_box = sorted(icx_2_best_anchor_box,                                   
                                    key=lambda x: predictions[ibx,x,icx_2_best_anchor_box[x]][0].item(), reverse=True)   ## (102)
                              retained_cells = sorted_icx_to_box[:5]                                                     ## (103)
                              objects_detected = []                                                                      ## (104)
                              for icx in retained_cells:                                                                 ## (105)
                                  pred_vec = predictions[ibx,icx, icx_2_best_anchor_box[icx]]                            ## (106)
                                  class_labels_predi  = pred_vec[-4:]                                                    ## (107)
                                  # print("test1", class_labels_predi)
                                  class_labels_probs = torch.nn.Softmax(dim=0)(class_labels_predi)                       ## (108)
                                  # print("test2", class_labels_probs)
                                  class_labels_probs = class_labels_probs[:-1]                                           ## (109)
                                  # print("test3", class_labels_probs)
                                  if torch.all(class_labels_probs < 0.1):                                               ## (110)
                                      predicted_class_label = None                                                       ## (111)
                                  else:                                                                                
                                      best_predicted_class_index = (class_labels_probs == class_labels_probs.max())      ## (112)
                                      best_predicted_class_index =torch.nonzero(best_predicted_class_index,as_tuple=True)## (113)
                                      predicted_class_label =self.rpg.class_labels[best_predicted_class_index[0].item()] ## (114)
                                      objects_detected.append(predicted_class_label)                                     ## (115)
                              print("[batch image=%d]  objects found in descending probability order: " % ibx, 
                                                                                                    objects_detected)     ## (116)
                              print("ground truth: ", bbox_label_tensor[0])                                                                     
                      if display_images:
                          logger = logging.getLogger()
                          old_level = logger.level
                          logger.setLevel(100)
                          plt.figure(figsize=[15,4])
                          plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                            padding=3, pad_value=255).cpu(), (1,2,0)))
                          plt.savefig("test2.png")
                          plt.show()
                          logger.setLevel(old_level)
          print("\nFinished Training\n")
          plt.figure(figsize=(10,5))
          plt.title("Loss vs. Iterations")
          plt.plot(Loss_tally)
          plt.xlabel("iterations")
          plt.ylabel("Loss")
          plt.legend()
          plt.savefig("training_loss.png")
          plt.show()
          torch.save(net.state_dict(), self.rpg.path_saved_yolo_model)
          return net
    
      # Python program to check if rectangles overlap
      # Code from https://www.geeksforgeeks.org/find-two-rectangles-overlap/
      class Point:
          def __init__(self, x, y):
              self.x = x
              self.y = y

      def doOverlap(self, l1, r1, l2, r2):       
        # To check if either rectangle is actually a line
          # For example  :  l1 ={-1,0}  r1={1,1}  l2={0,-1}  r2={0,1}         
        # if (l1.x == r1.x or l1.y == r1.y or l2.x == r2.x or l2.y == r2.y):
        #     # the line cannot have positive overlap
        #     return False      
        # If one rectangle is on left side of other
        if(l1.x >= r2.x or l2.x >= r1.x):
            return False    
        # If one rectangle is above other
        if(r1.y <= l2.y or r2.y <= l1.y):
            return False   
        return True

      def run_code_for_testing_multi_instance_detection(self, net, display_labels=False, display_images=False):        
            net.load_state_dict(torch.load(self.rpg.path_saved_yolo_model))
            net = net.to(self.rpg.device)
            yolo_interval = self.rpg.yolo_interval
            num_yolo_cells = (self.rpg.image_size[0] // yolo_interval) * (self.rpg.image_size[1] // yolo_interval)
            num_anchor_boxes =  5    # (height/width)   1/5  1/3  1/1  3/1  5/1
            yolo_tensor = torch.zeros( self.rpg.batch_size, num_yolo_cells, num_anchor_boxes, 8 )
            gt_bboxes={}
            pred_labels = []
            gt_lables = []
            false_alarm = 0
            total_detect_obj = 0

            with torch.no_grad():
                for iter, data in enumerate(self.test_dataloader):
                    im_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data
                    
                    im_tensor   = im_tensor.to(self.rpg.device)               
                    bbox_tensor = bbox_tensor.to(self.rpg.device)
                    bbox_label_tensor = bbox_label_tensor.to(self.rpg.device)
                    yolo_tensor = yolo_tensor.to(self.rpg.device)

                    output = net(im_tensor)
                    predictions = output.view(self.rpg.batch_size,num_yolo_cells,num_anchor_boxes,9)

                    for idx in range(5):
                        class_label_of_object = bbox_label_tensor[0,idx].item()
                        if class_label_of_object == 13: continue
                        i1 = int(bbox_tensor[0,idx,0])
                        i2 = int(bbox_tensor[0,idx,2])
                        j1 = int(bbox_tensor[0,idx,1])
                        j2 = int(bbox_tensor[0,idx,3])
                        im_tensor[0,0,j1:j2,i1] = 1
                        im_tensor[0,0,j1:j2,i2] = 1
                        im_tensor[0,0,j1,i1:i2] = 1
                        im_tensor[0,0,j2,i1:i2] = 1
                        l = self.Point(i1, j1)
                        r = self.Point(i2, j2)
                        gt_bboxes[idx] = [l, r, class_label_of_object]                  
    
                    for ibx in range(predictions.shape[0]):                             # for each batch image
                        icx_2_best_anchor_box = {ic : None for ic in range(36)}
                        for icx in range(predictions.shape[1]):                         # for each yolo cell
                            cell_predi = predictions[ibx,icx]               
                            prev_best = 0
                            for anchor_bdx in range(cell_predi.shape[0]):
                                if cell_predi[anchor_bdx][0] > cell_predi[prev_best][0]:
                                    prev_best = anchor_bdx
                            best_anchor_box_icx = prev_best   
                            icx_2_best_anchor_box[icx] = best_anchor_box_icx
                        sorted_icx_to_box = sorted(icx_2_best_anchor_box, 
                                    key=lambda x: predictions[ibx,x,icx_2_best_anchor_box[x]][0].item(), reverse=True)
                        retained_cells = sorted_icx_to_box[:5]
                    objects_detected = []
                    for icx in retained_cells:
                        pred_vec = predictions[ibx, icx, icx_2_best_anchor_box[icx]]
                        target_vec = bbox_tensor.tolist()
                        class_labels_predi  = pred_vec[-4:]                                             
                        class_labels_probs = torch.nn.Softmax(dim=0)(class_labels_predi)
                        class_labels_probs = class_labels_probs[:-1]
                        
                        if torch.all(class_labels_probs < 0.05): 
                            predicted_class_label = None
                        else:
                            best_predicted_class_index = (class_labels_probs == class_labels_probs.max())
                            best_predicted_class_index = torch.nonzero(best_predicted_class_index, as_tuple=True)
                            predicted_class_label = self.rpg.class_labels[best_predicted_class_index[0].item()]
                            objects_detected.append(predicted_class_label)

                            # draw detected objects
                            output_bb = pred_vec[1:5].tolist()
                            yolocell_center_i =  (icx % 6) * yolo_interval + float(yolo_interval) / 2.0                      ## (38)
                            yolocell_center_j =  math.floor(icx / 6) * yolo_interval + float(yolo_interval) / 2.0
                            k1 = int(yolocell_center_j + output_bb[1] * yolo_interval - output_bb[2] * yolo_interval / 2)
                            k2 = int(yolocell_center_j + output_bb[1] * yolo_interval + output_bb[2] * yolo_interval / 2)
                            l1 = int(yolocell_center_i + output_bb[0] * yolo_interval - output_bb[3] * yolo_interval / 2)
                            l2 = int(yolocell_center_i + output_bb[0] * yolo_interval + output_bb[3] * yolo_interval / 2)
                                                     
                            im_tensor[0,2,k1:k2,l1] = 1                      
                            im_tensor[0,2,k1:k2,l2] = 1
                            im_tensor[0,2,k1,l1:l2] = 1
                            im_tensor[0,2,k2,l1:l2] = 1

                            lp = self.Point(l1, k1)
                            rp = self.Point(l2, k2)
                            
                            overlap = False
                            for key, gt_bbox in gt_bboxes.items():
                              if self.doOverlap(gt_bbox[0], gt_bbox[1], lp, rp):
                                pred_labels += [best_predicted_class_index[0].item()]
                                gt_lables += [gt_bbox[2]]
                                overlap = True
                                if overlap: break

                            if overlap == False:
                              false_alarm += 1
                            total_detect_obj += 1

                    if iter % 5 == 4:
                        print("\n\n\n\nShowing output for test batch %d: " % (iter+1))        
                        print("[batch image=%d]  objects found in descending probability order: " % ibx, objects_detected)
                        print("ground truth: ", bbox_label_tensor[0])

                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[7,7])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True,
                                                                          padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.savefig("bbtest"+str(iter)+".png")
                        plt.show()
                        logger.setLevel(old_level)

                cf_matrix = confusion_matrix(gt_lables, pred_labels)
                accuracy = accuracy_score(gt_lables, pred_labels)
                xticklabels = self.rpg.class_labels
                yticklabels = self.rpg.class_labels
                plt.figure()
                ax = sns.heatmap(cf_matrix, annot=True, fmt="d", xticklabels=xticklabels, yticklabels=yticklabels)
                plt.title("YoloLikeNet" + "\n"+'Accuracy: '+str(accuracy) + "\n"+'False alarm rate: '+str(false_alarm/total_detect_obj))
                plt.xlabel('Prediction Label')
                plt.ylabel('Ground Truth Label')
                plt.tight_layout()

                plt.savefig("confusion.png") 
