#!/usr/bin/env python

##  one_neuron_classifier.py

"""
A one-neuron model is characterized by a single expression that you see in the value
supplied for the constructor parameter "expressions".  In the expression supplied, the
names that being with 'x' are the input variables and the names that begin with the
other letters of the alphabet are the learnable parameters.
"""

import random
import numpy as np
import operator
import matplotlib.pyplot as plt

seed = 0           
random.seed(seed)
np.random.seed(seed)

from ComputationalGraphPrimer import *

class ComputationalGraphPrimerSGDPlus(ComputationalGraphPrimer):
    def __init__(self, *args, **kwargs ):
        self.mu = 0;
        super(ComputationalGraphPrimerSGDPlus, self).__init__(*args, **kwargs )

    ######################################### one neuron model ###########################################
    def run_training_loop_one_neuron_model(self, mu=0):
        training_data = self.training_data
        self.vals_for_learnable_params = {param: random.uniform(0,1) for param in self.learnable_params}
        self.bias = random.uniform(0,1)
        self.prev_step = {param: 0 for param in self.learnable_params}
        self.bias_step = 0
        self.mu = mu

        class DataLoader:
            def __init__(self, training_data, batch_size):
                self.training_data = training_data
                self.batch_size = batch_size
                self.class_0_samples = [(item, 0) for item in self.training_data[0]]
                self.class_1_samples = [(item, 1) for item in self.training_data[1]]
            def __len__(self):
                return len(self.training_data[0]) + len(self.training_data[1])
            def _getitem(self):    
                cointoss = random.choice([0,1])
                if cointoss == 0:
                    return random.choice(self.class_0_samples)
                else:
                    return random.choice(self.class_1_samples)            
            def getbatch(self):
                batch_data,batch_labels = [],[]
                maxval = 0.0
                for _ in range(self.batch_size):
                    item = self._getitem()
                    if np.max(item[0]) > maxval: 
                        maxval = np.max(item[0])
                    batch_data.append(item[0])
                    batch_labels.append(item[1])
                batch_data = [item/maxval for item in batch_data]                
                batch = [batch_data, batch_labels]
                return batch                

        data_loader = DataLoader(self.training_data, batch_size=self.batch_size)
        loss_running_record = []
        i = 0
        avg_loss_over_literations = 0.0
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            y_preds, deriv_sigmoids =  self.forward_prop_one_neuron_model(data_tuples)
            loss = sum([(abs(class_labels[i] - y_preds[i]))**2 for i in range(len(class_labels))])
            loss_avg = loss / float(len(class_labels))
            avg_loss_over_literations += loss_avg
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_literations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_literations)
                print("[iter=%d]  loss = %.4f" %  (i+1, avg_loss_over_literations))
                avg_loss_over_literations = 0.0
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            deriv_sigmoid_avg = sum(deriv_sigmoids) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(map(operator.truediv, data_tuple_avg, 
                                     [float(len(class_labels))] * len(class_labels) ))
            self.backprop_and_update_params_one_neuron_model(y_error_avg, data_tuple_avg, deriv_sigmoid_avg)

        return loss_running_record
        # plt.figure()     
        # plt.plot(loss_running_record) 
        # plt.show()   

    ######################################################################################################
    def backprop_and_update_params_one_neuron_model(self, y_error, vals_for_input_vars, deriv_sigmoid):
        """
        As should be evident from the syntax used in the following call to backprop function,

           self.backprop_and_update_params_one_neuron_model( y_error_avg, data_tuple_avg, deriv_sigmoid_avg)
                                                                     ^^^             ^^^                ^^^
                                                               
        the values fed to the backprop function for its three arguments are averaged over the training 
        samples in the batch.  This in keeping with the spirit of SGD that calls for averaging the 
        information retained in the forward propagation over the samples in a batch.
        """
        input_vars = self.independent_vars
        vals_for_input_vars_dict =  dict(zip(input_vars, list(vals_for_input_vars)))
        vals_for_learnable_params = self.vals_for_learnable_params
        for i,param in enumerate(self.vals_for_learnable_params):
            step = self.mu * self.prev_step[param] + self.learning_rate * y_error * vals_for_input_vars_dict[input_vars[i]] * deriv_sigmoid
            self.prev_step[param] = step
            self.vals_for_learnable_params[param] += step
        self.bias_step = self.mu * self.bias_step + self.learning_rate * y_error * deriv_sigmoid
        self.bias += self.bias_step
    ######################################################################################################

cgp = ComputationalGraphPrimerSGDPlus(
               one_neuron_model = True,
               expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
               output_vars = ['xw'],
               dataset_size = 5000,
               learning_rate = 1e-3,
              # learning_rate = 5 * 1e-2,
               training_iterations = 40000,
               batch_size = 8,
               display_loss_how_often = 100,
               debug = True,
      )


cgp.parse_expressions()

#cgp.display_network1()
cgp.display_network2()

cgp.gen_training_data()

loss_running_record0 = cgp.run_training_loop_one_neuron_model(0)
loss_running_record1 = cgp.run_training_loop_one_neuron_model(0.99)

plt.figure()     
plt.plot(loss_running_record0, label = 'SGD') 
plt.plot(loss_running_record1, label = 'SGD+') 
plt.title('Loss vs iteration of SGD + and SGD for one neuron')
plt.xlabel('Iterations/100')
plt.ylabel('Loss')
plt.legend()
plt.savefig('one_neuron_loss.jpg')
plt.show()   

