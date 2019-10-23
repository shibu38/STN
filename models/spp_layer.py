import torch
import torch.nn as nn
import numpy as np

'''
https://github.com/Gary-Deeplearning/Pytorch-Spatial_Pyramid_Pool
'''

class Spatial_Pooling(nn.Module):
    def __init__(self, output_size, pooling_type='max'):
        '''
        output_size: the height and width after spp layer, tuple containing size of (h_out,w_out)
        pooling_type: the type of pooling , string 
        '''
        super(Spatial_Pooling, self).__init__()
        self.output_size = output_size
        self.pooling_type = pooling_type

    def forward(self, x):
        # print(x.size())
        batch_size, c, h, w  = x.size()

        assert self.output_size[0]<=h and self.output_size[1]<=w

        kernel_size=(h-self.output_size[0]+1,w-self.output_size[1]+1)

        if self.pooling_type == 'max':
            self.spp = nn.MaxPool2d(kernel_size=kernel_size,stride=1)
        else:
            self.spp = nn.AdaptiveAvgPool2d(kernel_size=kernel_size, stride=1)

        x= self.spp(x)
        # print(x.size())
        return x ##torch.Size([bacth_size, channels, h_out, w_out])

if __name__=="__main__":
    inp=torch.rand(10,3,500,20)
    spp=Spatial_Pooling(output_size=(10,10))
    out=spp(inp)
    # print(out.size())

    net=nn.Sequential(
            Spatial_Pooling(output_size=(10,10)),
            Spatial_Pooling(output_size=(5,5))
        )
    out2=net(inp)
    # print(out2.size())