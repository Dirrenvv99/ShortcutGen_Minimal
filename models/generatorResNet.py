from importlib.metadata import requires
from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import MaxPool1d
from einops import rearrange
from torch.autograd import Variable


###########################
# Generator: Resnet
###########################

# To control feature map in generator
ngf = 64

class GeneratorResnet(nn.Module):
    def __init__(self, inception=False, data_dim='high', eps = 8, no_tanh_bounding = False, feature_output = False):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        :param eps: epsilon bound value for the output
        '''
        super(GeneratorResnet, self).__init__()
        self.inception = inception
        self.data_dim = data_dim
        self.eps = eps
        self.no_tanh_bounding = no_tanh_bounding        
        self.feature_output = feature_output
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        if self.data_dim == 'high':
            self.resblock3 = ResidualBlock(ngf * 4)
            self.resblock4 = ResidualBlock(ngf * 4)
            self.resblock5 = ResidualBlock(ngf * 4)
            self.resblock6 = ResidualBlock(ngf * 4)
            # self.resblock7 = ResidualBlock(ngf*4)
            # self.resblock8 = ResidualBlock(ngf*4)
            # self.resblock9 = ResidualBlock(ngf*4)

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.data_dim == 'high':
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
        #TSNE here
        if self.feature_output:
            return x
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)
        if self.no_tanh_bounding:
            return x
        else:
            return torch.tanh(x) * self.eps/255 #Bounds the ouput to be between [-eps/255, eps/255] 


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual

class GeneratorResnetEnsemble(nn.Module):
    def __init__(self, p_number = 0, a_number = 0, n_number = 0, data_dim='low', maximum_combine = False, eps = 8, feature_output = False):
        assert(p_number + a_number + n_number > 1)
        super(GeneratorResnetEnsemble, self).__init__()
        for idx in range(p_number + n_number + a_number):
            setattr(self, f"Model_{idx+1}", GeneratorResnet(data_dim=data_dim, no_tanh_bounding=True, feature_output = feature_output))
        self.p_number = p_number
        self.n_number = n_number
        self.a_number = a_number
        self.eps = eps
        self.maximum_combine = maximum_combine
        self.conv_layer_p = nn.Conv2d(in_channels= 3 * p_number, out_channels= 3, kernel_size=3, stride=1, padding=1)
        self.conv_layer_an = nn.Conv2d(in_channels= 3 * (a_number + n_number), out_channels= 3, kernel_size=3, stride=1, padding=1)
        self.eval_pool = ChannelPool(kernel_size=3, stride=2, padding=1, dilation=1)

    def forward(self, *args):
        assert(len(args) == 1 or len(args) == 2)
    
        p_output = []
        an_output = []
        org_images = args[0]
        #if eval is true aug_images == orig_images, so this will still work
        aug_images = args[-1]
        #output generation of postaug
        if self.p_number > 0:
            for idx in range(self.p_number):
                p_output.append(getattr(self, f"Model_{idx+1}")(org_images))
        #output generation of aug
        if self.a_number > 0:
            for idx in range(self.p_number, self.p_number + self.a_number):
                an_output.append(getattr(self, f"Model_{idx+1}")(aug_images))
        #output generation of noaug
        if self.n_number > 0:
            for idx in range(self.p_number + self.a_number, self.p_number + self.a_number + self.n_number):
                an_output.append(getattr(self, f"Model_{idx+1}")(org_images))
        
        #concatenating outputs over the channels. p_size: B, 3 * p_number, H, W; an_size: B, 3 * (a_number  + n_number), H, W
        #combining with conv_layer the postaug outputs and the noaug and aug outputs seperatly
        if self.p_number > 1:
            p_out = torch.cat(p_output, dim = 1)
            p_out = self.conv_layer_p(p_out)
            p_out = torch.tanh(p_out) * self.eps/255
        elif self.p_number == 1:
            p_out = torch.Tensor(len(p_output), *p_output[0].size())
            p_out = torch.cat(p_output)
            p_out = torch.tanh(p_out) * self.eps/255
        if self.a_number + self.n_number > 1:
            an_out = torch.cat(an_output, dim = 1)
            an_out = self.conv_layer_an(an_out)
            an_out = torch.tanh(an_out) * self.eps/255
        elif self.a_number + self.n_number == 1:
            an_out = torch.Tensor(len(an_output), an_output[0].size())
            an_out = torch.cat(an_output)
            an_out = torch.tanh(an_out) * self.eps/255
        if self.training:
            if self.p_number > 0 and self.a_number + self.n_number > 0:
                return p_out, an_out
            elif self.p_number > 0:
                return p_out
            else:
                return an_out  
        else:            
            if self.p_number > 0 and self.a_number + self.n_number > 0:
                if self.maximum_combine:
                    out = torch.maximum(p_out, an_out)
                    return out
                else:
                    out = torch.cat((p_out, an_out), dim = 1)
                    return self.eval_pool(out)  
            elif self.p_number > 0:
                return p_out
            else:
                return an_out

class GeneratorResnet_P_Ensemble(nn.Module):
    def __init__(self, p_number = 0, data_dim='low', resolution = 32,  maximum_combine = False, eps = 8, finetuning = False, feature_output = False):
        assert(p_number > 2)
        super(GeneratorResnet_P_Ensemble, self).__init__()
        for idx in range(p_number):
            setattr(self, f"Model_{idx+1}", GeneratorResnet(data_dim=data_dim, feature_output = feature_output))
        self.p_number = p_number
        self.eps = eps
        self.maximum_combine = maximum_combine

        self.context_conv = nn.Conv2d(in_channels= 3, out_channels= 1, kernel_size=1)
        self.context_lin = nn.Linear(resolution*resolution, p_number)
        self.finetuning = finetuning
        self.flatten = nn.Flatten()
        self.SoftMax = nn.Softmax(dim = 1)

    def forward(self, input):   
        p_output = []
        #output generation of postaug
        for idx in range(self.p_number):
            p_output.append(getattr(self, f"Model_{idx+1}")(input).unsqueeze(dim = 0)) #list of 1 x B x C x H x W tensors

        p_output = torch.cat(p_output) # p_number x B x C x H x W

        if not self.finetuning and self.training:
            return torch.unbind(p_output)
        else:
            x = self.context_conv(input) # B x 3 (C) x H x W
            x  = self.flatten(x) # B x 1 x H x W -> B x (H * W)
            x = self.context_lin(x) # B x (H * W) -> B * p_number
            weights = self.SoftMax(x) 

            noise = torch.einsum('pbchw, bp -> bchw', p_output, weights)
            # noise_size = p_output.size()
            # noise = p_output.view(noise_size[1], noise_size[0], noise_size[2], noise_size[3], noise_size[4])
            # weights = weights.unsqueeze(2).unsqueeze(2).unsqueeze(2)
            # noise = noise*weights
            # noise = torch.sum(noise, dim =1)

            # weights_out = weights[:]



            #Given that we bound at the end, a softmax is actually not needed, changed this to be bounding at the start, thus now a softmax is needed
            # output =  # B x C x H x W

            # output = torch.tanh(output) * self.eps/255 # bound the output (now done before the finetuning loop)

            return noise
    def get_finetune_params(self):
        params = [
        {'params': self.context_conv.parameters()},
        {'params': self.context_lin.parameters()},
        {'params': self.flatten.parameters()},
        {'params': self.SoftMax.parameters()}
        ]
        return params

class CombinerModel(nn.Module):
    def __init__(self, p_number = 0, resolution = 32):
        assert(p_number > 0)
        super(CombinerModel, self).__init__()
        self.context_conv = nn.Conv2d(in_channels= 3, out_channels= 1, kernel_size=1)
        self.context_lin = nn.Linear(resolution*resolution, p_number)
        self.flatten = nn.Flatten()
        self.SoftMax = nn.Softmax(dim = 1)

    def forward(self, input):
        x = self.context_conv(input)    
        x = self.flatten(x)
        x = self.context_lin(x)
        x = self.SoftMax(x)
        return x
    



if __name__ == '__main__':
    netG = GeneratorResnet_P_Ensemble(50, data_dim='low')
    test_sample = torch.rand(1, 3, 32, 32)
    print('Generator output:', netG(test_sample).size())
    print('Generator parameters:', sum(p.numel() for p in netG.parameters() if p.requires_grad))