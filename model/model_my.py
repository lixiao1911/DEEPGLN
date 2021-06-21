"""created by L.X
"""
#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import model.resnet_my as resnet
import math

class LinearWise(nn.Module):
    def __init__(self, in_features, bias=True):
        super(LinearWise, self).__init__()
        self.in_features = in_features

        self.weight = nn.Parameter(torch.Tensor(self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        x = input * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x 

class background_resnet(nn.Module):
    def __init__(self, embedding_size, num_classes, backbone='resnet50'):
        super(background_resnet, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained models
        
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=False)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=False)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=False)
        elif backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=False)
        elif backbone == 'resnet34':
            self.pretrained = resnet.resnet34(pretrained=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
            

        self.linearwise1=LinearWise(256)
        self.linearwise2=LinearWise(256)
        self.linearwise3=LinearWise(256)

        self.fc0 = nn.Linear(768, embedding_size)
        self.bn0 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        self.last = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # input x: minibatch x 1 x 40 x 40
        # train: 64 x 1 x 40 x 100
        # enroll: 1 x 1 x 40 x 200
        #print('x:',x.shape)
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.layer1(x)

        x = self.pretrained.layer2(x)

        x = self.pretrained.layer3(x)


        
        out=torch.split(x,[3,4,3],dim=2)
        #out=torch.split(x,[w//3,w//3+w%3,w//3],dim=3)
        out1=out[0]
        out2=out[1]
        out3=out[2]
        
        out1=F.adaptive_avg_pool2d(out1,(1,1))
        out2=F.adaptive_avg_pool2d(out2,(1,1))
        out3=F.adaptive_avg_pool2d(out3,(1,1))
  
        out1=torch.squeeze(out1)
        out1=self.linearwise1(out1)
 

        out2=torch.squeeze(out2)
        out2=self.linearwise2(out2)

  
        out3=torch.squeeze(out3)
        out3=self.linearwise3(out3)

        
        
        out=torch.cat((out1,out2,out3),dim=0)#attention dim!!! different in train =1;enroll=0 
        
        out = torch.squeeze(out)
        out = out.view(x.size(0), -1)
        spk_embedding = self.fc0(out)
        out = F.relu(self.bn0(spk_embedding)) # [batch, n_embed]
        out = self.last(out)

        
        return spk_embedding, out    
