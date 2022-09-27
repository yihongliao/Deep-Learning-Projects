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

# dataset_archive_test =  "sentiment_dataset_test_3.tar.gz"
dataset_archive_test = "sentiment_dataset_test_200.tar.gz"

path_to_saved_embeddings = "/content/drive/MyDrive/HW8/"


dls = DLStudio(
                  dataroot = dataroot,
                  path_saved_model = "/content/drive/MyDrive/HW8/",
                  momentum = 0.9,
                  learning_rate =  1e-5,
                  epochs = 5,
                  batch_size = 1,
                  classes = ('negative','positive'),
                  use_gpu = True,
              )



text_cl = DLStudio.TextClassificationWithEmbeddings( dl_studio = dls )

dataserver_test = DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset(
                                 train_or_test = 'test',
                                 dl_studio = dls,
                                 dataset_file = dataset_archive_test,
                                 path_to_saved_embeddings = path_to_saved_embeddings,
                   )

text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset('test', dataserver_test)

nnGRU_model = text_cl.GRUnetWithEmbeddings(input_size=300, hidden_size=200, output_size=2, num_layers=1)
pmGRU_model = text_cl.pmGRUnetWithEmbeddings(input_size=300, hidden_size=200, output_size=2, num_layers=1)

# ## TESTING:
text_cl.run_code_for_testing_text_classification_with_GRU_word2vec(nnGRU_model, 'nnGRU_model')
text_cl.run_code_for_testing_text_classification_with_GRU_word2vec(pmGRU_model, 'pmGRU_model')


