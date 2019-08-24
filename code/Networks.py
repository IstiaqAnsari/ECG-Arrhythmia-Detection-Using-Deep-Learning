import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from math import floor,ceil

modlist = torch.nn.modules.container.ModuleList
L = 256 # Input length
kernel = 15
class Net(nn.Module):	
    def __init__(self,classes=18):
        print("Loaded")
        super(Net,self).__init__()
        self.classes = classes
        pad = self.padding_len(L,kernel,1)
        self.conv1 = nn.Conv1d(1,32,kernel_size=kernel,stride = 1,padding = pad)
        self.bnrelu1 = self.bn_relu(32)
        channel_start = 32
        channel = 32
        sub_sample_length = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        self.resnet = nn.ModuleList()
        for i,sub_sample in enumerate(sub_sample_length):
            self.resnet.append(self.res_block(i,channel,sub_sample))
            channel = 2**int(i/4)*32
            #break
        self.pool = nn.MaxPool1d(2)
        self.bnrelu2 = self.bn_relu(channel)
        self.linear1 = nn.Linear(256,self.classes)
        self.soft = nn.Softmax(dim=0)
    def forward(self,x):
        x = self.conv1(x)
        for lay in self.bnrelu1:
            x = lay(x)
        for i,lay in enumerate(self.resnet):
            y = x
            if(type(lay)==modlist):
                for la in lay:
                    if(type(la)==modlist):
                        for l in la:
                            x = l(x)
                    else:
                        x = la(x)
            else:x = lay(x)
            if(i%2):
                y = self.pool(y)
            if(i>0 and i%4==0):
                temp = torch.zeros_like(y)
                y = torch.cat([y,temp],1)
            x = x+y
        for lay in self.bnrelu2:
            x = lay(x)
        
        x = x.view(-1, self.num_flat_features(x))
        x = self.linear1(x)
        x = self.soft(x)
        return x
    def res_block(self,block_index,chan,subsample):
        layers = nn.ModuleList()
        if block_index>0:
            layers.append(self.bn_relu(chan))
        outchan = 2**int(block_index/4)*32
        pad = self.padding_len(L,kernel,subsample)
        layers.append(nn.Conv1d(chan,outchan,kernel_size=kernel,stride = subsample,padding = pad))
        layers.append(self.bn_relu(outchan))
        layers.append(nn.Dropout(.2))
        pad = self.padding_len(L,kernel,1)
        layers.append(nn.Conv1d(outchan,outchan,kernel_size=kernel,stride = 1,padding = pad))
        return layers
    
    def bn_relu(self,channels):
        layers = nn.ModuleList()
        layers.append(nn.BatchNorm1d(channels))
        layers.append(nn.ReLU())
        return layers
    def padding_len(self,i,k,s):
            return floor(kernel/2)
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
