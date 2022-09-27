#!/usr/bin/env python

##  text_classification_with_GRU_word2vec.py

"""
This script is an attempt at solving the sentiment classification problem
with an RNN that uses a GRU to get around the problem of vanishing gradients
that are common to neural networks with feedback.
"""

import random
import numpy
import torch
import os, sys
import matplotlib.pyplot as plt

"""
seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)
"""


##  watch -d -n 0.5 nvidia-smi

from MyDLStudio import *

dataroot = "/content/data/"

# dataset_archive_train = "sentiment_dataset_train_3.tar.gz"
dataset_archive_train = "sentiment_dataset_train_200.tar.gz"

path_to_saved_embeddings = "/content/drive/MyDrive/HW8/"


dls = DLStudio(
                  dataroot = dataroot,
                  path_saved_model = "/content/drive/MyDrive/HW8/",
                  momentum = 0.9,
                  learning_rate =  1e-5,
                  epochs = 1,
                  batch_size = 1,
                  classes = ('negative','positive'),
                  use_gpu = True,
              )



text_cl = DLStudio.TextClassificationWithEmbeddings( dl_studio = dls )

dataserver_train = DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset(
                                 train_or_test = 'train',
                                 dl_studio = dls,
                                 dataset_file = dataset_archive_train,
                                 path_to_saved_embeddings = path_to_saved_embeddings,
                   )

text_cl.dataserver_train = dataserver_train

text_cl.load_SentimentAnalysisDataset('train', dataserver_train)

nnGRU_model = text_cl.GRUnetWithEmbeddings(input_size=300, hidden_size=200, output_size=2, num_layers=1)
pmGRU_model = text_cl.pmGRUnetWithEmbeddings(input_size=300, hidden_size=200, output_size=2, num_layers=1)

number_of_learnable_params = sum(p.numel() for p in nnGRU_model.parameters() if p.requires_grad)

num_layers = len(list(nnGRU_model.parameters()))

print("\n\nThe number of layers in the nnGRU model: %d" % num_layers)
print("\nThe number of learnable parameters in the nnGRU model: %d" % number_of_learnable_params)

number_of_learnable_params = sum(p.numel() for p in pmGRU_model.parameters() if p.requires_grad)

num_layers = len(list(pmGRU_model.parameters()))

print("\n\nThe number of layers in the pmGRU model: %d" % num_layers)
print("\nThe number of learnable parameters in the pmGRU model: %d" % number_of_learnable_params)

## TRAINING:
print("\nStarting training\n")
nnGRU_training_loss_tally = text_cl.run_code_for_training_for_text_classification_with_GRU_word2vec(nnGRU_model, 'nnGRU_model', display_train_loss=True)
pmGRU_training_loss_tally = text_cl.run_code_for_training_for_text_classification_with_GRU_word2vec(pmGRU_model, 'pmGRU_model', display_train_loss=True)

plt.figure(figsize=(10,5))
plt.title("Training Loss vs. Iterations")
plt.plot(nnGRU_training_loss_tally, label = 'nnGRU')
plt.plot(pmGRU_training_loss_tally, label = 'pmGRU')
plt.xlabel("iterations")
plt.ylabel("training loss")
plt.legend()
plt.savefig("/content/drive/MyDrive/HW8/training_loss.png")
plt.show()

