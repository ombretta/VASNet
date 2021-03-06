__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import  *
from layer_norm import  *

import sys
sys.path.append("../../instructional_videos/i3d_breakfast/src/")

from i3dpt import I3D
from i3d_last_layer import I3D_after_maxPool3d

import subprocess

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


class i3d_afterMaxPool3d_SelfAttention(nn.Module):
    
    def __init__(self, i3d_input_interval=30):
        super(i3d_afterMaxPool3d_SelfAttention, self).__init__()

        self.i3d_input_interval = i3d_input_interval
        self.I3D_after_maxPool3d = I3D_after_maxPool3d(num_classes=400)
        self.VASNet = VASNet()

    def forward(self, x, seq_len):
        
        # print(get_gpu_memory_map())
        
        # print(x.shape)

        # timesteps = x.shape[2]
        # all_features = torch.zeros([math.ceil(timesteps/2), 1024], device=x.get_device())
        # i = 0
        
        # while i < timesteps:
            
        #     x_temp = x[:,:,int(i):int(i+int(8*1)/2*self.i3d_input_interval),:,:]
            
        #     print("i", i, "i/2", i/2)
            
        #     print("i", i, "i/4", i/4)
        #     print("t_temp", x_temp.shape)
            
        #     print(get_gpu_memory_map())
            
        #     features = self.I3D_after_maxPool3d.extract(x_temp)
            
        #     features = F.adaptive_avg_pool3d(features, (None, 1, 1))
        #     features = features.squeeze(3).squeeze(3).squeeze(0)
        #     features = features.permute(1,0)
        #     print("features", features.shape)

        #     all_features[round(i/2):round(i/2)+features.shape[0]] = features
            
        #     i += (8*1)/2*self.i3d_input_interval
            
        #     print(get_gpu_memory_map())
        
        features = self.I3D_after_maxPool3d.extract(x)
        features = F.adaptive_avg_pool3d(features, (None, 1, 1))
        features = features.squeeze(3).squeeze(3).squeeze(0)
        all_features = features.permute(1,0)
            
        # print("VASNet input", all_features.shape)
        y, att_weights_ = self.VASNet(all_features.unsqueeze(0), all_features.shape[1])

        return y, att_weights_
    

class i3d_SelfAttention(nn.Module):
    
    def __init__(self, i3d_input_interval=30):
        super(i3d_SelfAttention, self).__init__()

        self.i3d_input_interval = i3d_input_interval
        self.I3D = I3D(num_classes=400)
        self.VASNet = VASNet()

    def forward(self, x, seq_len):
        
        print(x.shape)

        timesteps = x.shape[1]
        all_features = torch.zeros([math.ceil(timesteps/8), 1024], device=x.get_device())
        i = 0
        
        while i < timesteps:
            
            x_temp = x[:,i:i+8*2*self.i3d_input_interval,:,:,:]
            
            x_temp = x_temp.permute(0, 4, 1, 2, 3)
            
            _, mixed_5c, _ = self.I3D.extract(x_temp)
            
            features = F.adaptive_avg_pool3d(mixed_5c, (None, 1, 1))
            features = features.squeeze(3).squeeze(3).squeeze(0)
            features = features.permute(1,0)
            all_features[round(i/8):round(i/8)+features.shape[0]] = features
            
            i += 8*2*self.i3d_input_interval
            
        y, att_weights_ = self.VASNet(all_features.unsqueeze(0), all_features.shape[1])

        return y, att_weights_
    
    

class SelfAttention(nn.Module):

    def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()

        self.apperture = apperture
        self.ignore_itself = ignore_itself

        self.m = input_size
        self.output_size = output_size

        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        self.drop50 = nn.Dropout(0.5)



    def forward(self, x):
        n = x.shape[0]  # sequence length

        K = self.K(x)  # ENC (n x m) => (n x H) H= hidden size
        Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size
        V = self.V(x)

        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1,0))

        if self.ignore_itself:
            # Zero the diagonal activations (a distance of each frame with itself)
            logits[torch.eye(n).byte()] = -float("Inf")

        if self.apperture > 0:
            # Set attention to zero to frames further than +/- apperture from the current one
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")

        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.drop50(att_weights_)
        y = torch.matmul(V.transpose(1,0), weights).transpose(1,0)
        y = self.output_linear(y)

        return y, att_weights_



class VASNet(nn.Module):

    def __init__(self):
        super(VASNet, self).__init__()

        self.m = 1024 # cnn features size
        self.hidden_size = 1024

        self.att = SelfAttention(input_size=self.m, output_size=self.m)
        self.ka = nn.Linear(in_features=self.m, out_features=1024)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)
        
        # Added standard torch batchnorm
        self.batch_norm_y = torch.nn.BatchNorm1d(self.m)
        self.batch_norm_ka = torch.nn.BatchNorm1d(self.ka.out_features)


    def forward(self, x, seq_len):

        m = x.shape[2] # Feature size

        # Place the video frames to the batch dimension to allow for batch arithm. operations.
        # Assumes input batch size = 1.
        x = x.view(-1, m)
        y, att_weights_ = self.att(x)

        y = y + x
        y = self.drop50(y)
        y = self.layer_norm_y(y)
        y = self.batch_norm_y(y) # added bn

        # Frame level importance score regression
        # Two layer NN
        y = self.ka(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)
        y = self.batch_norm_ka(y)

        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)

        return y, att_weights_



if __name__ == "__main__":
    pass
