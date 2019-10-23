import torch
from torch import nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.width = 30
        self.height = 40
        self.num_classes = 10
        self.inp_ch = 3
        self.num_fil = 64

        self.conv1 = nn.Conv2d(self.inp_ch, self.num_fil, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.num_fil, self.num_fil, 3, 1, 1)

        self.conv3 = nn.Conv2d(self.num_fil, self.num_fil * 2, 3, 1, 1)
        self.conv4 = nn.Conv2d(self.num_fil * 2, self.num_fil * 2, 3, 1, 1)

        self.conv5 = nn.Conv2d(self.num_fil * 2, self.num_fil * 3, 3, 1, 1)
        self.conv6 = nn.Conv2d(self.num_fil * 3, self.num_fil * 3, 3, 1, 1)
        self.conv7 = nn.Conv2d(self.num_fil * 3, self.num_fil * 3, 3, 1, 1)

        self.conv8 = nn.Conv2d(self.num_fil * 3, self.num_fil * 4, 3, 1, 1)
        self.conv9 = nn.Conv2d(self.num_fil * 4, self.num_fil * 4, 3, 1, 1)
        self.conv10 = nn.Conv2d(self.num_fil * 4, self.num_fil * 4, 3, 1, 1)

        self.conv11 = nn.Conv2d(self.num_fil * 4, self.num_fil * 5, 3, 1, 1)
        self.conv12 = nn.Conv2d(self.num_fil * 5, self.num_fil * 5, 3, 1, 1)
        self.conv13 = nn.Conv2d(self.num_fil * 5, self.num_fil * 5, 3, 1, 1)

        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(self.height * self.width * self.num_fil * 5, self.num_fil * 5)
        self.fc2 = nn.Linear(self.num_fil * 5, self.num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool1(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.maxpool1(x)

        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.maxpool1(x)

        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.maxpool1(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.fc2(x)

        output = F.softmax(x, dim=1)
        return output


if __name__ == "__main__":
    inp = torch.rand(10, 3, 30, 40)
    model = VGG16()
    out = model(inp)
    print(out.size())
