#A denoiser module for spatial denoising of calcium imaging neural data 
import os
import numpy as np

import torch
import torch.nn as nn
# from NID.models import DnCNN
from skimage import restoration



class denoiser():
    
    __supported = {"DnCNN"}
    
    
    def __init__(self, Method, DnCNNfile = "", DnCNNlayers = 17, DnCNNchannels = 1, DnCNNnoise = 0.1):
        """Initializes a denoiser object
        Args: 
            Method (string): Indicates the denoising method used the given denoising object. Current support for:
                1. "DnCNN": Neural Network denoising: this method uses a DnCNN
                2. NNfile (string): NNfile location relative to the current location. Must be .pth file. Example: NNfile = workspace/test/neuralnetwork.pth
                3. DnCNNlayers (int): The number of layers in the DnCNN being used
                4. DnCNNchannels (int): The number of channels in the CNN
                5. DnCNNnoise (double): The noise level upon which the DnCNN was originally trained
                    
        
        
        """
        
        #Verify method provided is correct
        assert Method in self.__supported, "Error While Initializing Denoiser: Unsupported Denoising Method Provided"
        self.Method = Method
        
        if Method=="DnCNN":
            assert os.path.exists(DnCNNfile), "Invalid filepath:{} provided for neural model".format(DnCNNfile)
            assert DnCNNfile.lower().endswith(".pth"), "Incorrect file type for neural model: .pth file expected"
            assert isinstance(DnCNNlayers, int) and DnCNNlayers > 0, "Invalid input for number of layers"
            assert isinstance(DnCNNchannels, int) and DnCNNchannels > 0, "Invalid number of channels"
            assert DnCNNnoise >= 0, "Invalid noise level"
            
            
        #Assign values to class object
        self.DnCNNfile = DnCNNfile
        self.DnCNNlayers = DnCNNlayers
        self.DnCNNchannels = DnCNNchannels
        self.DnCNNnoise = 0.1
        
    def denoise(self, img_raw):
        """Denoise the image using the various supported methods
        Args:
            img: a 2D image that needs to be denoised 
            
        Output:
            img_denoised: a 2D image that has been denoised
        """
        
        if self.Method == "DnCNN":
            #Denoise the image using the existing neural network
            model = DnCNN(channels=self.DnCNNchannels, num_of_layers=self.DnCNNlayers)
#             model = nn.DataParallel(net)# device_ids=device_ids)
            model.load_state_dict(torch.load(self.DnCNNfile,map_location=lambda storage, loc: storage))
            model.eval()

            #We can approximate the noise level below. In the future, we can do a more accurate estimation of Gaussian noise 
            noiselevel = img_raw.std()
#             noiselevel = restoration.estimate_sigma(img_raw)
        
            img_raw = img_raw[np.newaxis, np.newaxis, ...] / noiselevel * self.DnCNNnoise
            img_input = torch.Tensor(img_raw)
            img_output = model(img_input).detach().numpy()
            img_denoised = img_raw -img_output
                
            final_estimate = img_denoised * noiselevel / self.DnCNNnoise #Re-normalize the image
            final_estimate = final_estimate.squeeze()
                

        return final_estimate 
                
                
                
                      

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=5):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features,
                                kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels,
                                kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
