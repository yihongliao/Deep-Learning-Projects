import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
import numpy as np
from PIL import ImageFilter
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

#______________________________  DLStudio Class Definition  ________________________________

class DLStudio(object):

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
        if  path_saved_model:
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

    ########################################################################################
    ########  Start Definition of Inner Class TextClassificationWithEmbeddings  ############

    class TextClassificationWithEmbeddings(nn.Module):             
        def __init__(self, dl_studio,dataserver_train=None,dataserver_test=None,dataset_file_train=None,dataset_file_test=None):
            super(DLStudio.TextClassificationWithEmbeddings, self).__init__()
            self.dl_studio = dl_studio
            self.dataserver_train = dataserver_train
            self.dataserver_test = dataserver_test

        class SentimentAnalysisDataset(torch.utils.data.Dataset):
            """
            In relation to the SentimentAnalysisDataset defined for the TextClassification section of 
            DLStudio, the __getitem__() method of the dataloader must now fetch the embeddings from
            the word2vec word vectors.

            Class Path:  DLStudio -> TextClassificationWithEmbeddings -> SentimentAnalysisDataset
            """
            def __init__(self, dl_studio, train_or_test, dataset_file, path_to_saved_embeddings=None):
                super(DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset, self).__init__()
                import gensim.downloader as gen_api
#                self.word_vectors = gen_api.load("word2vec-google-news-300")
                self.path_to_saved_embeddings = path_to_saved_embeddings
                self.train_or_test = train_or_test
                root_dir = dl_studio.dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if path_to_saved_embeddings is not None:
                    import gensim.downloader as genapi
                    from gensim.models import KeyedVectors 
                    if os.path.exists(path_to_saved_embeddings + 'vectors.kv'):
                        self.word_vectors = KeyedVectors.load(path_to_saved_embeddings + 'vectors.kv')
                    else:
                        print("""\n\nSince this is your first time to install the word2vec embeddings, it may take"""
                              """\na couple of minutes. The embeddings occupy around 3.6GB of your disk space.\n\n""")
                        self.word_vectors = genapi.load("word2vec-google-news-300")               
                        ##  'kv' stands for  "KeyedVectors", a special datatype used by gensim because it 
                        ##  has a smaller footprint than dict
                        self.word_vectors.save(path_to_saved_embeddings + 'vectors.kv')    
                if train_or_test == 'train':
                    if sys.version_info[0] == 3:
                        self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset)
                    self.categories = sorted(list(self.positive_reviews_train.keys()))
                    self.category_sizes_train_pos = {category : len(self.positive_reviews_train[category]) for category in self.categories}
                    self.category_sizes_train_neg = {category : len(self.negative_reviews_train[category]) for category in self.categories}
                    self.indexed_dataset_train = []
                    for category in self.positive_reviews_train:
                        for review in self.positive_reviews_train[category]:
                            self.indexed_dataset_train.append([review, category, 1])
                    for category in self.negative_reviews_train:
                        for review in self.negative_reviews_train[category]:
                            self.indexed_dataset_train.append([review, category, 0])
                    random.shuffle(self.indexed_dataset_train)
                elif train_or_test == 'test':
                    if sys.version_info[0] == 3:
                        self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset)
                    self.vocab = sorted(self.vocab)
                    self.categories = sorted(list(self.positive_reviews_test.keys()))
                    self.category_sizes_test_pos = {category : len(self.positive_reviews_test[category]) for category in self.categories}
                    self.category_sizes_test_neg = {category : len(self.negative_reviews_test[category]) for category in self.categories}
                    self.indexed_dataset_test = []
                    for category in self.positive_reviews_test:
                        for review in self.positive_reviews_test[category]:
                            self.indexed_dataset_test.append([review, category, 1])
                    for category in self.negative_reviews_test:
                        for review in self.negative_reviews_test[category]:
                            self.indexed_dataset_test.append([review, category, 0])
                    random.shuffle(self.indexed_dataset_test)

            def review_to_tensor(self, review):
                list_of_embeddings = []
                for i,word in enumerate(review):
                    if word in self.word_vectors.key_to_index:
                        embedding = self.word_vectors[word]
                        list_of_embeddings.append(np.array(embedding))
                    else:
                        next
                review_tensor = torch.FloatTensor( list_of_embeddings )
                return review_tensor

            def sentiment_to_tensor(self, sentiment):
                """
                Sentiment is ordinarily just a binary valued thing.  It is 0 for negative
                sentiment and 1 for positive sentiment.  We need to pack this value in a
                two-element tensor.
                """        
                sentiment_tensor = torch.zeros(2)
                if sentiment == 1:
                    sentiment_tensor[1] = 1
                elif sentiment == 0: 
                    sentiment_tensor[0] = 1
                sentiment_tensor = sentiment_tensor.type(torch.long)
                return sentiment_tensor

            def __len__(self):
                if self.train_or_test == 'train':
                    return len(self.indexed_dataset_train)
                elif self.train_or_test == 'test':
                    return len(self.indexed_dataset_test)

            def __getitem__(self, idx):
                sample = self.indexed_dataset_train[idx] if self.train_or_test == 'train' else self.indexed_dataset_test[idx]
                review = sample[0]
                review_category = sample[1]
                review_sentiment = sample[2]
                review_sentiment = self.sentiment_to_tensor(review_sentiment)
                review_tensor = self.review_to_tensor(review)
                category_index = self.categories.index(review_category)
                sample = {'review'       : review_tensor, 
                          'category'     : category_index, # should be converted to tensor, but not yet used
                          'sentiment'    : review_sentiment }
                return sample

        def load_SentimentAnalysisDataset(self, train_or_test, dataserver):  
            if train_or_test == 'train': 
              self.train_dataloader = torch.utils.data.DataLoader(dataserver,
                        batch_size=self.dl_studio.batch_size,shuffle=True, num_workers=0)
            elif train_or_test == 'test':
              self.test_dataloader = torch.utils.data.DataLoader(dataserver,
                                batch_size=self.dl_studio.batch_size,shuffle=False, num_workers=0)

        class pmGRU(nn.Module):
            """
            This GRU implementation is based primarily on a "Minimal Gated" version of a GRU as described in
            "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks" by Joel Heck and Fathi 
            Salem. The Wikipedia page on "Gated_recurrent_unit" has a summary presentation of the equations 
            proposed by Heck and Salem.
            """
            def __init__(self, input_size, hidden_size, output_size): 
                super(DLStudio.TextClassificationWithEmbeddings.pmGRU, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size
                ## for forget gate:
                self.project1 = nn.Sequential( nn.Linear(self.input_size + self.hidden_size, self.hidden_size), nn.Sigmoid() )
                ## for interim out:
                self.project2 = nn.Sequential( nn.Linear( self.input_size + self.hidden_size, self.hidden_size), nn.Tanh() ) 
                ## for final out
                self.project3 = nn.Sequential( nn.Linear( self.hidden_size, self.output_size ), nn.Tanh() )                   
        
            def forward(self, x, h, sequence_end=False):
                combined1 = torch.cat((x, h), 2)
                forget_gate = self.project1(combined1)  
                interim =  forget_gate * h
                combined2  = torch.cat((x, interim), 2)
                output_interim =  self.project2( combined2 )
                output = (1 - forget_gate) * h  +  forget_gate * output_interim
                if sequence_end == False:
                    return output, output
                else:
                    final_out = self.project3(output)
                    return final_out, final_out

        class GRUnetWithEmbeddings(nn.Module):
            """
            For this embeddings adapted version of the GRUnet shown earlier, we can assume that
            the 'input_size' for a tensor representing a word is always 300.
            Source: https://blog.floydhub.com/gru-with-pytorch/
            with the only modification that the final output of forward() is now
            routed through LogSoftmax activation. 

            Class Path:  DLStudio -> TextClassificationWithEmbeddings -> GRUnetWithEmbeddings 
            """
            def __init__(self, input_size, hidden_size, output_size, num_layers=1): 
                """
                -- input_size is the size of the tensor for each word in a sequence of words.  If you word2vec
                       embedding, the value of this variable will always be equal to 300.
                -- hidden_size is the size of the hidden state in the RNN
                -- output_size is the size of output of the RNN.  For binary classification of 
                       input text, output_size is 2.
                -- num_layers creates a stack of GRUs
                """
                super(DLStudio.TextClassificationWithEmbeddings.GRUnetWithEmbeddings, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.gru = nn.GRU(input_size, hidden_size, num_layers)
                self.fc = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
                self.logsoftmax = nn.LogSoftmax(dim=1)
                self.batch_size = 1
                
            def forward(self, x, h):
                out, h = self.gru(x, h)
                out = self.fc(self.relu(out[:,-1]))
                out = self.logsoftmax(out)
                return out, h

            def init_hidden(self):
                weight = next(self.parameters()).data
                #                  num_layers  batch_size    hidden_size
                hidden = weight.new(self.num_layers, self.batch_size,  self.hidden_size    ).zero_()
                return hidden

        # Task 2
        class pmGRUnetWithEmbeddings(nn.Module):
            """
            For this embeddings adapted version of the GRUnet shown earlier, we can assume that
            the 'input_size' for a tensor representing a word is always 300.
            Source: https://blog.floydhub.com/gru-with-pytorch/
            with the only modification that the final output of forward() is now
            routed through LogSoftmax activation. 

            Class Path:  DLStudio -> TextClassificationWithEmbeddings -> pmGRUnetWithEmbeddings 
            """
            def __init__(self, input_size, hidden_size, output_size, num_layers=1): 
                """
                -- input_size is the size of the tensor for each word in a sequence of words.  If you word2vec
                       embedding, the value of this variable will always be equal to 300.
                -- hidden_size is the size of the hidden state in the RNN
                -- output_size is the size of output of the RNN.  For binary classification of 
                       input text, output_size is 2.
                -- num_layers creates a stack of GRUs
                """
                super(DLStudio.TextClassificationWithEmbeddings.pmGRUnetWithEmbeddings, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.gru = DLStudio.TextClassificationWithEmbeddings.pmGRU(input_size, hidden_size, output_size)
                self.fc = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
                self.logsoftmax = nn.LogSoftmax(dim=1)
                self.batch_size = 1
                
            def forward(self, x, h):
                out, h = self.gru(x, h)
                out = self.fc(self.relu(out[:,-1]))
                out = self.logsoftmax(out)
                return out, h

            def init_hidden(self):
                weight = next(self.parameters()).data
                #                  num_layers  batch_size    hidden_size
                hidden = weight.new(self.num_layers, self.batch_size,  self.hidden_size    ).zero_()
                return hidden
                
        def save_model(self, model, model_name):
            "Save the trained model to a disk file"
            torch.save(model.state_dict(), self.dl_studio.path_saved_model + model_name)

        def run_code_for_training_for_text_classification_with_GRU_word2vec(self, net, model_name, display_train_loss=False): 
            filename_for_out = model_name + "_performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE = open(filename_for_out, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            ##  Note that the GREnet now produces the LogSoftmax output:
            criterion = nn.NLLLoss()
            accum_times = []
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            training_loss_tally = []
            start_time = time.perf_counter()
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    review_tensor = review_tensor.to(self.dl_studio.device)
                    sentiment = sentiment.to(self.dl_studio.device)
                    ## The following type conversion needed for MSELoss:
                    ##sentiment = sentiment.float()
                    optimizer.zero_grad()
                    hidden = net.init_hidden().to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                    loss = criterion(output, torch.argmax(sentiment, 1))
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    if i % 200 == 199:    
                        avg_loss = running_loss / float(200)
                        training_loss_tally.append(avg_loss)
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%d  iter:%4d  elapsed_time:%4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                        accum_times.append(current_time-start_time)
                        FILE.write("%.5f\n" % avg_loss)
                        FILE.flush()
                        running_loss = 0.0
            self.save_model(net, model_name)
            print("Total Training Time: {}".format(str(sum(accum_times))))
            print("\nFinished Training\n\n")
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig("/content/drive/MyDrive/HW8/training_loss_" + model_name + ".png")
                plt.show()
            return training_loss_tally

        def run_code_for_testing_text_classification_with_GRU_word2vec(self, net, model_name):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model + model_name))
            classification_accuracy = 0.0
            negative_total = 0
            positive_total = 0
            confusion_matrix = torch.zeros(2,2)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    hidden = net.init_hidden()
                    for k in range(review_tensor.shape[1]):
                        output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                    predicted_idx = torch.argmax(output).item()
                    gt_idx = torch.argmax(sentiment).item()
                    if i % 100 == 99:
                        print("   [i=%d]    predicted_label=%d       gt_label=%d" % (i+1, predicted_idx,gt_idx))
                    if predicted_idx == gt_idx:
                        classification_accuracy += 1
                    if gt_idx == 0: 
                        negative_total += 1
                    elif gt_idx == 1:
                        positive_total += 1
                    confusion_matrix[gt_idx,predicted_idx] += 1
            print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
            out_percent = np.zeros((2,2), dtype='float')
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            print("\n\nNumber of positive reviews tested: %d" % positive_total)
            print("\n\nNumber of negative reviews tested: %d" % negative_total)
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                      "
            out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
            print(out_str + "\n")
            for i,label in enumerate(['true negative', 'true positive']):
                out_str = "%12s:  " % label
                for j in range(2):
                    out_str +=  "%18s%%" % out_percent[i,j]
                print(out_str)