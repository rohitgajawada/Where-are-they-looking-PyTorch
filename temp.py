import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

alex = models.alexnet(pretrained=True)
alex2 = models.alexnet(pretrained=True)

class AlexSal(nn.Module):
    def __init__(self):
        super(AlexSal, self).__init__()
        self.features = nn.Sequential(
            *list(alex.features.children())[:-3]
        )

        self.relu = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(256) #do i add batchnorm?
        self.conv6 = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.bn5(self.relu(self.features(x)))
        x = self.relu(self.conv6(x))
        return x

class AlexGaze(nn.Module):
    def __init__(self):
        super(AlexGaze, self).__init__()
        self.features = nn.Sequential(
            *list(alex2.features.children())[:-3]
        )

        self.relu = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(256) #do i add batchnorm?

    def forward(self, x):
        x = self.bn5(self.relu(self.features(x)))
        return x

salmodel = AlexSal()
gazemodel = AlexGaze()
x = Variable(torch.randn(1, 3, 227, 227))
print(gazemodel(x))
