import torch
from torch import nn
import torch.nn.functional as F

from models.spp_layer import Spatial_Pooling

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception(nn.Module):
    __constants__ = ['branch2', 'branch3', 'branch4']

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.inceptionA = Inception(3,10,10,20,10,20,10)
        self.conv1=nn.Conv2d(in_channels=60,out_channels=80,kernel_size=2,stride=2,padding=1)
        self.conv2=nn.Conv2d(in_channels=80,out_channels=100,kernel_size=2,stride=2,padding=1)
        # self.conv3=nn.Conv2d(in_channels=20,out_channels=40,kernel_size=2,stride=1,padding=0)
        
        self.spp1=Spatial_Pooling(output_size=(2,2))
        self.spp2=Spatial_Pooling(output_size=(1,1))
        self.fc1=nn.Linear((100*2*2)+(100*1*1),150)
        self.fc2=nn.Linear(150,10)

        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.2)
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=2)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=5,kernel_size=2,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=5,out_channels=10,kernel_size=2,stride=1,padding=1),
            Spatial_Pooling(output_size=(3,3)),
            nn.ReLU(True)
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )


    # Spatial transformer network forward function
    def stn(self, x):
        # print(x.size())
        xs = self.localization(x)   #torch.Size([64, 10, 3, 3])
        # print(xs.size())
        xs = xs.view(-1, 10 * 3 * 3)    #torch.Size([64, 90])
        theta = self.fc_loc(xs)    #torch.Size([64, 6])
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        batch_size,_,_,_=x.size()
        x=self.inceptionA(x)
        # x=self.stn(x)
        # print(x.size())
        x=self.relu(self.maxpool1(self.conv1(x)))
        # print(x.size())
        x=self.relu(self.maxpool1(self.conv2(x)))
        # print(x.size())
        # x=self.relu(self.maxpool1(self.conv3(x)))
        # print(x.size())
        x=torch.cat([self.spp1(x).view(batch_size,-1),self.spp2(x).view(batch_size,-1)],dim=1)
        x=self.dropout(x)
        x=self.dropout(self.relu(self.fc1(x)))
        x=self.fc2(x)
        output=F.softmax(x, dim=1)
        return output