import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
import numpy as np
from PIL import ImageFilter, Image
import numbers
import re
import math
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import time
import logging
from pycocotools.coco import COCO
import requests
import json

#______________________________  MyDLStudio Class Definition  ________________________________

class MyDLStudio(object):

  def __init__(self, *args, **kwargs ):
    if args:
      raise ValueError(  
              '''DLStudio constructor can only be called with keyword arguments for 
                the following keywords: epochs, learning_rate, batch_size, momentum,
                convo_layers_config, image_size, dataroot, path_saved_model, classes, 
                image_size, convo_layers_config, fc_layers_config, debug_train, use_gpu, and 
                debug_test''')
    learning_rate = epochs = batch_size = convo_layers_config = momentum = None
    image_size = fc_layers_config = dataroot =  path_saved_model = classes = use_gpu = None
    debug_train  = debug_test = None
    if 'dataroot' in kwargs                      :   dataroot = kwargs.pop('dataroot')
    if 'learning_rate' in kwargs                 :   learning_rate = kwargs.pop('learning_rate')
    if 'momentum' in kwargs                      :   momentum = kwargs.pop('momentum')
    if 'epochs' in kwargs                        :   epochs = kwargs.pop('epochs')
    if 'batch_size' in kwargs                    :   batch_size = kwargs.pop('batch_size')
    if 'convo_layers_config' in kwargs           :   convo_layers_config = kwargs.pop('convo_layers_config')
    if 'image_size' in kwargs                    :   image_size = kwargs.pop('image_size')
    if 'fc_layers_config' in kwargs              :   fc_layers_config = kwargs.pop('fc_layers_config')
    if 'path_saved_model' in kwargs              :   path_saved_model = kwargs.pop('path_saved_model')
    if 'classes' in kwargs                       :   classes = kwargs.pop('classes') 
    if 'use_gpu' in kwargs                       :   use_gpu = kwargs.pop('use_gpu') 
    if 'debug_train' in kwargs                   :   debug_train = kwargs.pop('debug_train') 
    if 'debug_test' in kwargs                    :   debug_test = kwargs.pop('debug_test') 
    if len(kwargs) != 0: raise ValueError('''You have provided unrecognizable keyword args''')
    if dataroot:
      self.dataroot = dataroot
    if convo_layers_config:
      self.convo_layers_config = convo_layers_config
    if image_size:
      self.image_size = image_size
    if fc_layers_config:
      self.fc_layers_config = fc_layers_config
#            if fc_layers_config[0] is not -1:
      if fc_layers_config[0] != -1:
        raise Exception("""\n\n\nYour 'fc_layers_config' construction option is not correct. """
                        """The first element of the list of nodes in the fc layer must be -1 """
                        """because the input to fc will be set automatically to the size of """
                        """the final activation volume of the convolutional part of the network""")
    if path_saved_model:
      self.path_saved_model = path_saved_model
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
    self.debug_config = 0
#        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_gpu is False else "cpu")

  class DetectAndLocalize(nn.Module):             
    """
    The purpose of this inner class is to focus on object detection in images --- as
    opposed to image classification.  Most people would say that object detection
    is a more challenging problem than image classification because, in general,
    the former also requires localization.  The simplest interpretation of what
    is meant by localization is that the code that carries out object detection
    must also output a bounding-box rectangle for the object that was detected.

    You will find in this inner class some examples of LOADnet classes meant
    for solving the object detection and localization problem.  The acronym
    "LOAD" in "LOADnet" stands for

                "LOcalization And Detection"

    The different network examples included here are LOADnet1, LOADnet2, and
    LOADnet3.  For now, only pay attention to LOADnet2 since that's the class I
    have worked with the most for the 1.0.7 distribution.

    Class Path:   DLStudio  ->  DetectAndLocalize
    """
    def __init__(self, dl_studio, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
      super(MyDLStudio.DetectAndLocalize, self).__init__()
      self.dl_studio = dl_studio
      self.dataserver_train = dataserver_train
      self.dataserver_test = dataserver_test
      self.debug = False

    class cocoDataset(torch.utils.data.Dataset):
      """
      Class Path:   DLStudio  ->  DetectAndLocalize  ->  cocoDataset
      """
      def __init__(self, dl_studio, train_or_test, coco_json_path, images_per_class):
        super(MyDLStudio.DetectAndLocalize.cocoDataset, self).__init__()
        self.class_list = dl_studio.class_labels
        self.images_per_class = images_per_class
        self.coco_json_path = coco_json_path
        self.dl_studio = dl_studio
        self.data_path = self.dl_studio.dataroot + train_or_test + '/'
        self.label_list = []
        self.bbox_list = []
        self.image_r_list = []
        self.image_g_list = []
        self.image_b_list = []
        self.dataset_list = []
        self.class_labels = dl_studio.class_labels

        if train_or_test == 'train':         
          if os.path.isfile(self.coco_json_path + "list_train2017.json"):
            print('Loading training data json file...')
            self.dataset_list = json.load(open(self.coco_json_path + "list_train2017.json"))                    
          else:
            # initialize COCO api for instance annotations
            self.coco = COCO(self.coco_json_path + 'instances_train2017.json')
            print('Downloading training data images')
            self.dataset_list = self.download_images(train_or_test)
            print('Saving training data json file')
            json.dump(self.dataset_list, open(self.coco_json_path + "list_train2017.json", 'w'))

          print("Number of training data: ", len(self.dataset_list))

        else:    
          if os.path.isfile(self.coco_json_path + "list_val2017.json"):
            print('Loading testing data json file...')
            self.dataset_list = json.load(open(self.coco_json_path + "list_val2017.json"))                    
          else:
            self.coco = COCO(self.coco_json_path + 'instances_val2017.json')
            print('Downloading testing data images')
            self.dataset_list = self.download_images(train_or_test)
            print('Saving testing data json file')
            json.dump(self.dataset_list, open(self.coco_json_path + "list_val2017.json", 'w'))

          print("Number of testing data: ", len(self.dataset_list))
          
        
      def download_images(self, train_or_test):
        dataset_list = []
        for category in self.class_list:
          category_folder = self.data_path + category
          encodedClass = self.class_list.index(category)

          # create category folders
          if not os.path.exists(category_folder):
            os.makedirs(category_folder)

          catId = self.coco.getCatIds(catNms=category);
          imgId = self.coco.getImgIds(catIds=catId)
          print('Image pool: ', len(imgId))
          img_dicts = self.coco.loadImgs(imgId)

          i = 0
          j = 0
          while (i < self.images_per_class):
            img_dict = img_dicts[j]
            w0 = img_dict['width']
            h0 = img_dict['height']
            annId = self.coco.getAnnIds(imgIds=imgId[j], iscrowd=False)
            anns = self.coco.loadAnns(annId)
            bbox_area_list = []
            for ann in anns:
              bbox_area_list.append(ann['bbox'][2]*ann['bbox'][3])
            ann_dom_idx = np.argmax(bbox_area_list)          
            ann_dom = anns[ann_dom_idx]

            if train_or_test == 'train':
              is_valid_image = (ann_dom['category_id'] == catId[0]) and (ann_dom['bbox'][2]/w0 > 1/3 and ann_dom['bbox'][3]/h0 > 1/3)
            else:
              is_valid_image = (ann_dom['category_id'] == catId[0])
            j += 1

            if (is_valid_image):
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

              img_path = os.path.join(category_folder, img_name)
              with open(img_path, 'wb') as file:
                file.write(img_response.content)
              im = Image.open(img_path)
              if im.mode != "RGB":
                im = im.convert(mode="RGB")
              w_scale = (self.dl_studio.image_size[0]-1)/(w0-1)
              h_scale = (self.dl_studio.image_size[1]-1)/(h0-1)
              im_resized = im.resize(self.dl_studio.image_size, Image.BOX)
              np_im_resized = np.array(im_resized)

              im_resized.save(img_path)

              bbox_x0 = (ann_dom['bbox'][0] + 1 - 1)*w_scale + 1 - 1
              bbox_x1 = ((ann_dom['bbox'][0] + 1 + ann_dom['bbox'][2] - 1) - 1)*w_scale + 1 - 1
              bbox_y0 = (ann_dom['bbox'][1] + 1 - 1)*h_scale + 1 - 1
              bbox_y1 = ((ann_dom['bbox'][1] + 1 + ann_dom['bbox'][3] - 1) - 1)*h_scale + 1 - 1

              data_dict = {}
              data_dict['r'] = np_im_resized[:,:,0].tolist()
              data_dict['g'] = np_im_resized[:,:,1].tolist()
              data_dict['b'] = np_im_resized[:,:,2].tolist()
              data_dict['label'] = encodedClass
              data_dict['bbox'] = [bbox_x0, bbox_y0, bbox_x1, bbox_y1]
              dataset_list.append(data_dict)

              i+=1
              if i % 50 == 49:
                print(i, "/", self.images_per_class)

        return dataset_list
      
      def __len__(self):
        return len(self.dataset_list)
            
      def __getitem__(self, index):
        r = np.array(self.dataset_list[index]['r'])
        g = np.array(self.dataset_list[index]['g'])
        b = np.array(self.dataset_list[index]['b'])
        im_tensor = torch.zeros(3, self.dl_studio.image_size[0], self.dl_studio.image_size[1], dtype=torch.float)
        im_tensor[0,:,:] = torch.from_numpy(r)
        im_tensor[1,:,:] = torch.from_numpy(g)
        im_tensor[2,:,:] = torch.from_numpy(b)
        bb_tensor = torch.tensor(self.dataset_list[index]['bbox'], dtype=torch.float)
        sample = {'image': im_tensor, 'bbox': bb_tensor, 'label': self.dataset_list[index]['label']}
        return sample

    def load_coco_training_dataset(self, dataserver_train):       
      self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                          batch_size=self.dl_studio.batch_size, shuffle=True, num_workers=2)
    def load_coco_testing_dataset(self, dataserver_test): 
      self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.dl_studio. batch_size, shuffle=False, num_workers=2)

    class MySkipBlock(nn.Module):
      """
      Class Path:   DLStudio  ->  DetectAndLocalize  ->  SkipBlock
      """
      def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
        super(MyDLStudio.DetectAndLocalize.MySkipBlock, self).__init__()
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

    class MyLOADnet(nn.Module):
      """
      The acronym 'LOAD' stands for 'LOcalization And Detection'.
      LOADnet2 uses both convo and linear layers for regression

      Class Path:   MyDLStudio  ->  DetectAndLocalize  ->  MyLOADnet
      """ 
      def __init__(self, skip_connections=True, depth=8):
        super(MyDLStudio.DetectAndLocalize.MyLOADnet, self).__init__()
        if depth not in [8,10,12,14,16]:
          sys.exit("MyLOADnet has only been tested for 'depth' values 8, 10, 12, 14, and 16")
        self.depth = depth // 2
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
#                self.pool = nn.MaxPool2d(2, 2)
        self.bn1  = nn.BatchNorm2d(64)
        self.bn2  = nn.BatchNorm2d(128)
        self.skip64_arr = nn.ModuleList()
        for i in range(self.depth):
          self.skip64_arr.append(MyDLStudio.DetectAndLocalize.MySkipBlock(64, 64, skip_connections=skip_connections))
        self.skip64ds = MyDLStudio.DetectAndLocalize.MySkipBlock(64, 64, downsample=True, skip_connections=skip_connections)
        self.skip64to128 = MyDLStudio.DetectAndLocalize.MySkipBlock(64, 128, skip_connections=skip_connections)
        self.skip128_arr = nn.ModuleList()
        for i in range(self.depth):
          self.skip128_arr.append(MyDLStudio.DetectAndLocalize.MySkipBlock(128, 128, skip_connections=skip_connections))
        self.skip128ds = MyDLStudio.DetectAndLocalize.MySkipBlock(128, 128, downsample=True, skip_connections=skip_connections)
        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1000, 5)

        ##  for regression
        self.conv_seqn = nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),

          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),

          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
          nn.ReLU(inplace=True)

          # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
          # nn.ReLU(inplace=True),
          # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
          # nn.ReLU(inplace=True)
        )
        self.fc_seqn = nn.Sequential(
          nn.Linear(16384, 1024),
          nn.ReLU(inplace=True),
          nn.Linear(1024, 512),
          nn.ReLU(inplace=True),
          nn.Linear(512, 4)        ## output for the 4 coords (x_min,y_min,x_max,y_max) of BBox
        )

      def forward(self, x):
        x = nn.MaxPool2d(2,2)(torch.nn.functional.relu(self.conv(x)))          
        ## The labeling section:
        x1 = x.clone()
        for i,skip64 in enumerate(self.skip64_arr[:self.depth//4]):
          x1 = skip64(x1)     

        x1 = skip64(x1) 

        x1 = self.skip64ds(x1)
        for i,skip64 in enumerate(self.skip64_arr[self.depth//4:]):
          x1 = skip64(x1)                
        x1 = self.bn1(x1)
        x1 = self.skip64to128(x1)
        for i,skip128 in enumerate(self.skip128_arr[:self.depth//4]):
          x1 = skip128(x1)      

        x1 = skip128(x1) 

        x1 = self.bn2(x1)
        x1 = self.skip128ds(x1)
        for i,skip128 in enumerate(self.skip128_arr[self.depth//4:]):
          x1 = skip128(x1)                
        x1 = x1.view(-1, 2048 )
        x1 = torch.nn.functional.relu(self.fc1(x1))
        x1 = self.fc2(x1)
        ## The Bounding Box regression:
        x2 = self.conv_seqn(x)
        # flatten
        x2 = x2.view(x.size(0), -1)
        x2 = self.fc_seqn(x2)
        return x1,x2

    def run_code_for_training_with_CrossEntropy_and_MSE_Losses(self, net):        
      filename_for_out1 = "/content/drive/MyDrive/HW5/performance_numbers_" + str(self.dl_studio.epochs) + "label.txt"
      filename_for_out2 = "/content/drive/MyDrive/HW5/performance_numbers_" + str(self.dl_studio.epochs) + "regres.txt"
      FILE1 = open(filename_for_out1, 'w')
      FILE2 = open(filename_for_out2, 'w')
      net = copy.deepcopy(net)
      net = net.to(self.dl_studio.device)
      criterion1 = nn.CrossEntropyLoss()
      criterion2 = nn.MSELoss()
      optimizer = optim.SGD(net.parameters(), lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
      print("\n\nStarting training loop...\n\n")
      start_time = time.perf_counter()
      labeling_loss_tally = []   
      regression_loss_tally = [] 
      elapsed_time = 0.0   
      for epoch in range(self.dl_studio.epochs):  
        print("")
        running_loss_labeling = 0.0
        running_loss_regression = 0.0       
        for i, data in enumerate(self.train_dataloader):
          gt_too_small = False
          inputs, bbox_gt, labels = data['image'], data['bbox'], data['label']
          if i % 250 == 249:
            current_time = time.perf_counter()
            elapsed_time = current_time - start_time
            print("\n\n\n[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]      Ground Truth:     " % 
                      (epoch+1, self.dl_studio.epochs, i+1, elapsed_time) 
                    + ' '.join('%10s' % self.dataserver_train.class_labels[labels[j].item()] 
                                                      for j in range(self.dl_studio.batch_size)))
          inputs = inputs.to(self.dl_studio.device)
          labels = labels.to(self.dl_studio.device)
          bbox_gt = bbox_gt.to(self.dl_studio.device)
          optimizer.zero_grad()
          # if self.debug:
          #   self.dl_studio.display_tensor_as_image(
          #     torchvision.utils.make_grid(inputs.cpu(), nrow=4, normalize=True, padding=2, pad_value=10))
          outputs = net(inputs)
          outputs_label = outputs[0]
          bbox_pred = outputs[1]
          if i % 250 == 249:
            inputs_copy = inputs.detach().clone()
            inputs_copy = inputs_copy.cpu()
            bbox_pc = bbox_pred.detach().clone()
            bbox_pc[bbox_pc<0] = 0
            bbox_pc[bbox_pc>31] = 31
            bbox_pc[torch.isnan(bbox_pc)] = 0
            _, predicted = torch.max(outputs_label.data, 1)
            print("[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]  Predicted Labels:     " % 
                    (epoch+1, self.dl_studio.epochs, i+1, elapsed_time)  
                  + ' '.join('%10s' % self.dataserver_train.class_labels[predicted[j].item()] 
                                                      for j in range(self.dl_studio.batch_size)))
            for idx in range(self.dl_studio.batch_size):
              i1 = int(bbox_gt[idx][1])
              i2 = int(bbox_gt[idx][3])
              j1 = int(bbox_gt[idx][0])
              j2 = int(bbox_gt[idx][2])
              k1 = int(bbox_pc[idx][1])
              k2 = int(bbox_pc[idx][3])
              l1 = int(bbox_pc[idx][0])
              l2 = int(bbox_pc[idx][2])
              print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
              print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
              inputs_copy[idx,0,i1:i2,j1] = 255
              inputs_copy[idx,0,i1:i2,j2] = 255
              inputs_copy[idx,0,i1,j1:j2] = 255
              inputs_copy[idx,0,i2,j1:j2] = 255
              inputs_copy[idx,2,k1:k2,l1] = 255                      
              inputs_copy[idx,2,k1:k2,l2] = 255
              inputs_copy[idx,2,k1,l1:l2] = 255
              inputs_copy[idx,2,k2,l1:l2] = 255
          loss_labeling = criterion1(outputs_label, labels)
          loss_labeling.backward(retain_graph=True)        
          loss_regression = criterion2(bbox_pred, bbox_gt)
          loss_regression.backward()
          optimizer.step()
          running_loss_labeling += loss_labeling.item()    
          running_loss_regression += loss_regression.item()                
          if i % 250 == 249:    
            avg_loss_labeling = running_loss_labeling / float(250)
            avg_loss_regression = running_loss_regression / float(250)
            labeling_loss_tally.append(avg_loss_labeling)  
            regression_loss_tally.append(avg_loss_regression)    
            print("[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]       loss_labelling %.3f        loss_regression: %.3f " %  (epoch+1, self.dl_studio.epochs, i+1, elapsed_time, avg_loss_labeling, avg_loss_regression))
            FILE1.write("%.3f\n" % avg_loss_labeling)
            FILE1.flush()
            FILE2.write("%.3f\n" % avg_loss_regression)
            FILE2.flush()
            running_loss_labeling = 0.0
            running_loss_regression = 0.0
          if i % 250 == 249:
            logger = logging.getLogger()
            old_level = logger.level
            logger.setLevel(100)
            plt.figure(figsize=[8,3])
            plt.imshow(np.transpose(torchvision.utils.make_grid(inputs_copy, normalize=True,
                                                              padding=3, pad_value=255).cpu(), (1,2,0)))
            plt.savefig("/content/drive/MyDrive/HW5/bbox.png")
            plt.show()
            logger.setLevel(old_level)
      print("\nFinished Training\n")
      self.save_model(net)
      plt.figure(figsize=(10,5))
      plt.title("Labeling Loss vs. Iterations")
      plt.plot(labeling_loss_tally)
      plt.xlabel("iterations")
      plt.ylabel("labeling loss")
      plt.legend()
      plt.savefig("/content/drive/MyDrive/HW5/labeling_loss.png")
      plt.show()
      plt.title("regression Loss vs. Iterations")
      plt.plot(regression_loss_tally)
      plt.xlabel("iterations")
      plt.ylabel("regression loss")
      plt.legend()
      plt.savefig("/content/drive/MyDrive/HW5/regression_loss.png")
      plt.show()


    def save_model(self, model):
      '''
      Save the trained model to a disk file
      '''
      torch.save(model.state_dict(), self.dl_studio.path_saved_model)

    def run_code_for_testing_detection_and_localization(self, net):
      net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
      correct = 0
      total = 0
      confusion_matrix = torch.zeros(len(self.dataserver_test.class_labels), 
                                      len(self.dataserver_test.class_labels))
      class_correct = [0] * len(self.dataserver_test.class_labels)
      class_total = [0] * len(self.dataserver_test.class_labels)
      with torch.no_grad():
        for i, data in enumerate(self.test_dataloader):
          images, bounding_box, labels = data['image'], data['bbox'], data['label']
          labels = labels.tolist()
          if self.dl_studio.debug_test and i % 50 == 0:
            print("\n\n[i=%d:] Ground Truth:     " %i + ' '.join('%10s' % 
              self.dataserver_test.class_labels[labels[j]] for j in range(self.dl_studio.batch_size)))
          outputs = net(images)
          outputs_label = outputs[0]
          outputs_regression = outputs[1]
          outputs_regression[outputs_regression < 0] = 0
          outputs_regression[outputs_regression > 31] = 31
          outputs_regression[torch.isnan(outputs_regression)] = 0
          output_bb = outputs_regression.tolist()
          _, predicted = torch.max(outputs_label.data, 1)
          predicted = predicted.tolist()
          if self.dl_studio.debug_test and i % 50 == 0:
            print("[i=%d:] Predicted Labels: " %i + ' '.join('%10s' % 
                  self.dataserver_test.class_labels[predicted[j]] for j in range(self.dl_studio.batch_size)))
            for idx in range(self.dl_studio.batch_size):
              i1 = int(bounding_box[idx][1])
              i2 = int(bounding_box[idx][3])
              j1 = int(bounding_box[idx][0])
              j2 = int(bounding_box[idx][2])
              k1 = int(output_bb[idx][1])
              k2 = int(output_bb[idx][3])
              l1 = int(output_bb[idx][0])
              l2 = int(output_bb[idx][2])
              print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
              print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
              images[idx,0,i1:i2,j1] = 255
              images[idx,0,i1:i2,j2] = 255
              images[idx,0,i1,j1:j2] = 255
              images[idx,0,i2,j1:j2] = 255
              images[idx,2,k1:k2,l1] = 255                      
              images[idx,2,k1:k2,l2] = 255
              images[idx,2,k1,l1:l2] = 255
              images[idx,2,k2,l1:l2] = 255
            logger = logging.getLogger()
            old_level = logger.level
            logger.setLevel(100)
            plt.figure(figsize=[8,3])
            plt.imshow(np.transpose(torchvision.utils.make_grid(images, normalize=True,
                                                              padding=3, pad_value=255).cpu(), (1,2,0)))
            plt.savefig("/content/drive/MyDrive/HW5/valbbox.png")
            plt.show()
            logger.setLevel(old_level)
          for label,prediction in zip(labels,predicted):
            confusion_matrix[label][prediction] += 1
          total += len(labels)
          correct +=  [predicted[ele] == labels[ele] for ele in range(len(predicted))].count(True)
          comp = [predicted[ele] == labels[ele] for ele in range(len(predicted))]
          for j in range(self.dl_studio.batch_size):
            label = labels[j]
            class_correct[label] += comp[j]
            class_total[label] += 1
      print("\n")
      for j in range(len(self.dataserver_test.class_labels)):
          print('Prediction accuracy for %5s : %2d %%' % (
        self.dataserver_test.class_labels[j], 100 * class_correct[j] / class_total[j]))
      print("\n\n\nOverall accuracy of the network on the 1000 test images: %d %%" % 
                                                              (100 * correct / float(total)))
      print("\n\nDisplaying the confusion matrix:\n")
      out_str = "                "
      for j in range(len(self.dataserver_test.class_labels)):  
                            out_str +=  "%15s" % self.dataserver_test.class_labels[j]   
      print(out_str + "\n")
      for i,label in enumerate(self.dataserver_test.class_labels):
        out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                          for j in range(len(self.dataserver_test.class_labels))]
        out_percents = ["%.2f" % item.item() for item in out_percents]
        out_str = "%12s:  " % self.dataserver_test.class_labels[i]
        for j in range(len(self.dataserver_test.class_labels)): 
                                                out_str +=  "%15s" % out_percents[j]
        print(out_str)



