from DLStudio import DLStudio

import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvtF
import torch.optim as optim
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import glob                                                                                                           
import imageio       

#______________________________  AdversarialLearning Class Definition  ________________________________

class AdversarialLearning(object):

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''AdversarialLearning constructor can only be called with keyword arguments for the following
                      keywords: epochs, learning_rate, batch_size, momentum, image_size, dataroot, path_saved_model, 
                      use_gpu, latent_vector_size, ngpu, dlstudio, device, LAMBDA, clipping_threshold, and beta1''')
        allowed_keys = 'dataroot','image_size','path_saved_model','momentum','learning_rate','epochs','batch_size', \
                       'classes','use_gpu','latent_vector_size','ngpu','dlstudio', 'beta1', 'LAMBDA', 'clipping_threshold'
        keywords_used = kwargs.keys()                                                                 
        for keyword in keywords_used:                                                                 
            if keyword not in allowed_keys:                                                           
                raise SyntaxError(keyword + ":  Wrong keyword used --- check spelling")  
        learning_rate = epochs = batch_size = convo_layers_config = momentum = None
        image_size = fc_layers_config = dataroot =  path_saved_model = classes = use_gpu = None
        latent_vector_size = ngpu = beta1 = LAMBDA = clipping_threshold = None
        if 'latent_vector_size' in kwargs            :   latent_vector_size = kwargs.pop('latent_vector_size')
        if 'ngpu' in kwargs                          :   ngpu  = kwargs.pop('ngpu')           
        if 'dlstudio' in kwargs                      :   dlstudio  = kwargs.pop('dlstudio')
        if 'beta1' in kwargs                         :   beta1  = kwargs.pop('beta1')
        if 'LAMBDA' in kwargs                        :   LAMBDA  = kwargs.pop('LAMBDA')
        if 'clipping_threshold' in kwargs            :   clipping_threshold = kwargs.pop('clipping_threshold')
        if latent_vector_size:
            self.latent_vector_size = latent_vector_size
        if ngpu:
            self.ngpu = ngpu
        if dlstudio:
            self.dlstudio = dlstudio
        if beta1:
            self.beta1 = beta1
        if LAMBDA:
            self.LAMBDA = LAMBDA
        if clipping_threshold:
            self.clipping_threshold = clipping_threshold 

    ####################################################################################################
    ########################     Start Definition of Inner Class DataModeling     ######################
    ####################################################################################################

    class DataModeling(nn.Module):             
        """
        The purpose of this class is demonstrate adversarial learning for constructing a neural-network
        based data model.  After you have constructed such a model for a dataset, it should be possible
        to create an instance of the model starting with just a noise vector.  When this idea is 
        applied to images, what that means is that you should be able to create an image that looks 
        very much like those in the training data.  Since the inputs to the neural network for 
        generating such "fakes" is pure noise, each instance you create in this manner would be different
        and yet look very much like what was in the training dataset.

        Class Path:  AdversarialLearning  ->   DataModeling
        """
        def __init__(self, dlstudio, adversarial, num_workers=2):
            super(AdversarialLearning.DataModeling, self).__init__()
            self.dlstudio = dlstudio
            self.adversarial  = adversarial
            self.num_workers = num_workers
            self.train_dataloader = None
            self.device = torch.device("cuda:0" if (torch.cuda.is_available() and adversarial.ngpu > 0) else "cpu")

        def show_sample_images_from_dataset(self, dlstudio):
            data = next(iter(self.train_dataloader))    
            real_batch = data[0]
            first_im = real_batch[0]
            self.dlstudio.display_tensor_as_image(torchvision.utils.make_grid(real_batch, padding=2, pad_value=1, normalize=True))

        def set_dataloader(self):
            dataset = torchvision.datasets.ImageFolder(root=self.dlstudio.dataroot,       
                           transform = tvt.Compose([                 
                                                tvt.Resize(self.dlstudio.image_size),             
                                                tvt.CenterCrop(self.dlstudio.image_size),         
                                                tvt.ToTensor(),                     
                                                tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),         
                           ]))
            self.train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.dlstudio.batch_size, 
                                                                                     shuffle=True, num_workers=self.num_workers)

        def weights_init(self,m):        
            """
            Uses the DCGAN initializations for the weights
            """
            classname = m.__class__.__name__     
            if classname.find('Conv') != -1:         
                nn.init.normal_(m.weight.data, 0.0, 0.02)      
            elif classname.find('BatchNorm') != -1:         
                nn.init.normal_(m.weight.data, 1.0, 0.02)       
                nn.init.constant_(m.bias.data, 0)   
                
        #####################################   My Discriminator-Generator MyDG   ######################################
        class MyDiscriminator(nn.Module):
            """
            Class Path:  AdversarialLearning  ->   DataModeling  ->  MyDiscriminator
            """
            def __init__(self):
                super(AdversarialLearning.DataModeling.MyDiscriminator, self).__init__()
                self.conv1 = nn.Conv2d(3, 128, 3) ## (A)
                self.conv2 = nn.Conv2d(128, 128, 3) ## (B)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(128*14*14, 1000) ## (C)
                self.fc2 = nn.Linear(1000, 1)
                self.sig = nn.Sigmoid()
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x))) ## (D)
                x = x.view(-1, 128*14*14) ## (E)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                x = self.sig(x)
                return x

        #####################################   Discriminator-Generator DG1   ######################################
        class DiscriminatorDG1(nn.Module):
            """
            This is an implementation of the DCGAN Discriminator. I refer to the DCGAN network topology as
            the 4-2-1 network.  Each layer of the Discriminator network carries out a strided
            convolution with a 4x4 kernel, a 2x2 stride and a 1x1 padding for all but the final
            layer. The output of the final convolutional layer is pushed through a sigmoid to yield
            a scalar value as the final output for each image in a batch.

            Class Path:  AdversarialLearning  ->   DataModeling  ->  DiscriminatorDG1
            """
            def __init__(self):
                super(AdversarialLearning.DataModeling.DiscriminatorDG1, self).__init__()
                self.conv_in = nn.Conv2d(  3,    64,      kernel_size=4,      stride=2,    padding=1)
                self.conv_in2 = nn.Conv2d( 64,   128,     kernel_size=4,      stride=2,    padding=1)
                self.conv_in3 = nn.Conv2d( 128,  256,     kernel_size=4,      stride=2,    padding=1)
                self.conv_in4 = nn.Conv2d( 256,  512,     kernel_size=4,      stride=2,    padding=1)
                self.conv_in5 = nn.Conv2d( 512,  1,       kernel_size=4,      stride=1,    padding=0)
                self.bn1  = nn.BatchNorm2d(128)
                self.bn2  = nn.BatchNorm2d(256)
                self.bn3  = nn.BatchNorm2d(512)
                self.sig = nn.Sigmoid()
            def forward(self, x):                 
                x = torch.nn.functional.leaky_relu(self.conv_in(x), negative_slope=0.2, inplace=True)
                x = self.bn1(self.conv_in2(x))
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
                x = self.bn2(self.conv_in3(x))
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
                x = self.bn3(self.conv_in4(x))
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.2, inplace=True)
                x = self.conv_in5(x)
                x = self.sig(x)
                return x

        class GeneratorDG1(nn.Module):
            """
            This is an implementation of the DCGAN Generator. As was the case with the Discriminator network,
            you again see the 4-2-1 topology here.  A Generator's job is to transform a random noise
            vector into an image that is supposed to look like it came from the training dataset. (We refer 
            to the images constructed from noise vectors in this manner as fakes.)  As you will see later 
            in the "run_gan_code()" method, the starting noise vector is a 1x1 image with 100 channels.  In 
            order to output 64x64 output images, the network shown below use the Transpose Convolution 
            operator nn.ConvTranspose2d with a stride of 2.  If (H_in, W_in) are the height and the width 
            of the image at the input to a nn.ConvTranspose2d layer and (H_out, W_out) the same at the 
            output, the size pairs are related by
                         H_out   =   (H_in - 1) * s   +   k   -   2 * p
                         W_out   =   (W_in - 1) * s   +   k   -   2 * p
            
            were s is the stride and k the size of the kernel.  (I am assuming square strides, kernels, and 
            padding). Therefore, each nn.ConvTranspose2d layer shown below doubles the size of the input.

            Class Path:  AdversarialLearning  ->   DataModeling  ->  GeneratorDG1
            """
            def __init__(self):
                super(AdversarialLearning.DataModeling.GeneratorDG1, self).__init__()
                self.latent_to_image = nn.ConvTranspose2d( 100,   512,  kernel_size=4, stride=1, padding=0, bias=False)
                self.upsampler2 = nn.ConvTranspose2d( 512, 256, kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler3 = nn.ConvTranspose2d (256, 128, kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler4 = nn.ConvTranspose2d (128, 64,  kernel_size=4, stride=2, padding=1, bias=False)
                self.upsampler5 = nn.ConvTranspose2d(  64,  3,  kernel_size=4, stride=2, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(512)
                self.bn2 = nn.BatchNorm2d(256)
                self.bn3 = nn.BatchNorm2d(128)
                self.bn4 = nn.BatchNorm2d(64)
                self.tanh  = nn.Tanh()
            def forward(self, x):                     
                x = self.latent_to_image(x)
                x = torch.nn.functional.relu(self.bn1(x))
                x = self.upsampler2(x)
                x = torch.nn.functional.relu(self.bn2(x))
                x = self.upsampler3(x)
                x = torch.nn.functional.relu(self.bn3(x))
                x = self.upsampler4(x)
                x = torch.nn.functional.relu(self.bn4(x))
                x = self.upsampler5(x)
                x = self.tanh(x)
                return x
        ########################################   DG1 Definition ENDS   ###########################################

        ############################################################################################################
        ##  The training routines follow, first for a GAN constructed using either the DG1 and or the DG2 
        ##  Discriminator-Generator Networks, and then for a WGAN constructed using either the CG1 or the CG2
        ##  Critic-Generator Networks.
        ############################################################################################################

        def run_gan_code(self, dlstudio, adversarial, discriminator, generator, results_dir):
            """
            This function is meant for training a Discriminator-Generator based Adversarial Network.  
            The implementation shown uses several programming constructs from the "official" DCGAN 
            implementations at the PyTorch website and at GitHub. 

            Regarding how to set the parameters of this method, see the following script

                         dcgan_DG1.py

            in the "ExamplesAdversarialLearning" directory of the distribution.
            """
            dir_name_for_results = results_dir
            if os.path.exists(dir_name_for_results):
                files = glob.glob(dir_name_for_results + "/*")
                for file in files:
                    if os.path.isfile(file):
                        os.remove(file)
                    else:
                        files = glob.glob(file + "/*")
                        list(map(lambda x: os.remove(x), files))
            else:
                os.mkdir(dir_name_for_results)
            #  Set the number of channels for the 1x1 input noise vectors for the Generator:
            nz = 100
            netD = discriminator.to(self.device)
            netG = generator.to(self.device)
            #  Initialize the parameters of the Discriminator and the Generator networks according to the
            #  definition of the "weights_init()" method:
            netD.apply(self.weights_init)
            netG.apply(self.weights_init)
            #  We will use a the same noise batch to periodically check on the progress made for the Generator:
            fixed_noise = torch.randn(self.dlstudio.batch_size, nz, 1, 1, device=self.device)          
            #  Establish convention for real and fake labels during training
            real_label = 1   
            fake_label = 0         
            #  Adam optimizers for the Discriminator and the Generator:
            optimizerD = optim.Adam(netD.parameters(), lr=dlstudio.learning_rate, betas=(adversarial.beta1, 0.999))    
            optimizerG = optim.Adam(netG.parameters(), lr=dlstudio.learning_rate, betas=(adversarial.beta1, 0.999))
            #  Establish the criterion for measuring the loss at the output of the Discriminator network:
            criterion = nn.BCELoss()
            #  We will use these lists to store the results accumulated during training:
            img_list = []                               
            G_losses = []                               
            D_losses = []                               
            iters = 0                                   
            print("\n\nStarting Training Loop...\n\n")      
            start_time = time.perf_counter()            
            for epoch in range(dlstudio.epochs):        
                g_losses_per_print_cycle = []           
                d_losses_per_print_cycle = []           
                # For each batch in the dataloader
                for i, data in enumerate(self.train_dataloader, 0):         
                    #  As indicated in the DCGAN part of the doc section at the beginning of this file, the GAN
                    #  training boils down to carrying out a max-min optimization. Each iterative step
                    #  of the max part results in updating the Discriminator parameters and each iterative 
                    #  step of the min part results in the updating of the Generator parameters.  For each 
                    #  batch of the training data, we first do max and then do min.  Since the max operation 
                    #  affects both terms of the criterion shown in the doc section, it has two parts: In the
                    #  first part we apply the Discriminator to the training images using 1.0 as the target; 
                    #  and, in the second part, we supply to the Discriminator the output of the Generator 
                    #  and use -1.0 as the target. In what follows, the Discriminator is being applied to 
                    #  the training images:
                    netD.zero_grad()    
                    real_images_in_batch = data[0].to(self.device)     
                    #  Need to know how many images we pulled in since at the tailend of the dataset, the 
                    #  number of images may not equal the user-specified batch size:
                    b_size = real_images_in_batch.size(0)  
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)  
                    output = netD(real_images_in_batch).view(-1)  
                    lossD_for_reals = criterion(output, label)                                                   
                    lossD_for_reals.backward()                                                                   
            
                    #  That brings us the second part of what it takes to carry out the max operation on the
                    #  DCGAN criterion shown in the doc section at the beginning of this file.  This part
                    #  calls for applying the Discriminator to the images produced by the Generator from noise:
                    noise = torch.randn(b_size, nz, 1, 1, device=self.device)    
                    fakes = netG(noise) 
                    label.fill_(fake_label) 
                    output = netD(fakes.detach()).view(-1)  
                    lossD_for_fakes = criterion(output, label)    
                    lossD_for_fakes.backward()          
                    lossD = lossD_for_reals + lossD_for_fakes    
                    d_losses_per_print_cycle.append(lossD)  
                    optimizerD.step()  
            
                    #  That brings to the min part of the max-min optimization described in the doc section
                    #  at the beginning of this file.  The min part requires that we minimize "1 - D(G(z))"
                    #  which, since D is constrained to lie in the interval (0,1), requires that we maximize
                    #  D(G(z)).  We accomplish that by applying the Discriminator to the output of the 
                    #  Generator and use 1 as the target for each image:
                    netG.zero_grad()   
                    label.fill_(real_label)  
                    output = netD(fakes).view(-1)   
                    lossG = criterion(output, label)          
                    g_losses_per_print_cycle.append(lossG) 
                    lossG.backward()    
                    optimizerG.step() 
                    
                    if i % 100 == 99:                                                                           
                        current_time = time.perf_counter()                                                      
                        elapsed_time = current_time - start_time                                                
                        mean_D_loss = torch.mean(torch.FloatTensor(d_losses_per_print_cycle))                   
                        mean_G_loss = torch.mean(torch.FloatTensor(g_losses_per_print_cycle))                   
                        print("[epoch=%d/%d   iter=%4d   elapsed_time=%5d secs]     mean_D_loss=%7.4f    mean_G_loss=%7.4f" % 
                                      ((epoch+1),dlstudio.epochs,(i+1),elapsed_time,mean_D_loss,mean_G_loss))   
                        d_losses_per_print_cycle = []                                                           
                        g_losses_per_print_cycle = []                                                           
                    G_losses.append(lossG.item())                                                                
                    D_losses.append(lossD.item())                                                                
                    if (iters % 500 == 0) or ((epoch == dlstudio.epochs-1) and (i == len(self.train_dataloader)-1)):   
                        with torch.no_grad():             
                            fake = netG(fixed_noise).detach().cpu()  
                        img_list.append(torchvision.utils.make_grid(fake, padding=1, pad_value=1, normalize=True))
                    iters += 1              

            #  At the end of training, make plots from the data in G_losses and D_losses:
            plt.figure(figsize=(10,5))    
            plt.title("Generator and Discriminator Loss During Training")    
            plt.plot(G_losses,label="G")    
            plt.plot(D_losses,label="D") 
            plt.xlabel("iterations")   
            plt.ylabel("Loss")         
            plt.legend()          
            plt.savefig(dir_name_for_results + "/gen_and_disc_loss_training.png") 
            plt.show()    
            #  Make an animated gif from the Generator output images stored in img_list:            
            images = []           
            for imgobj in img_list:  
                img = tvtF.to_pil_image(imgobj)  
                img = np.array(img)
                images.append(img) 
            imageio.mimsave(dir_name_for_results + "/generation_animation.gif", images, fps=5)
            #  Make a side-by-side comparison of a batch-size sampling of real images drawn from the
            #  training data and what the Generator is capable of producing at the end of training:
            real_batch = next(iter(self.train_dataloader)) 
            real_batch = real_batch[0]
            plt.figure(figsize=(15,15))  
            plt.subplot(1,2,1)   
            plt.axis("off")   
            plt.title("Real Images")    
            plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch.to(self.device), 
                                                   padding=1, pad_value=1, normalize=True).cpu(),(1,2,0)))  
            plt.subplot(1,2,2)                                                                             
            plt.axis("off")                                                                                
            plt.title("Fake Images")                                                                       
            plt.imshow(np.transpose(img_list[-1],(1,2,0)))                                                 
            plt.savefig(dir_name_for_results + "/real_vs_fake_images.png")                                 
            plt.show()