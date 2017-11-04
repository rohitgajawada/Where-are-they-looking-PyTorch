import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

places_alex = torch.load('../whole_alexnet_places365.pth.tar')
imagenet_alex = models.alexnet(pretrained=True)
#LRN present in previous models but not here
#try out batchnorm and dropout

class AlexSal(nn.Module):
    def __init__(self):
        super(AlexSal, self).__init__()
        self.features = nn.Sequential(
            *list(places_alex.features.children())[:-2]
        )

        self.relu = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.bn5(self.relu(self.features(x)))
        x = self.relu(self.conv6(x))
        x = x.squeeze(1)
        return x

class AlexGaze(nn.Module):
    def __init__(self):
        super(AlexGaze, self).__init__()
        self.features = nn.Sequential(
            *list(imagenet_alex.features.children())
        )

        self.relu = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(9216, 500)
        self.fc2 = nn.Linear(669, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 169)
        self.sig = nn.Sigmoid()

        self.finalconv = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x, egrid):
        x = self.bn5(self.features(x))
        x = x.view(-1, 9216)
        x = self.relu(self.fc1(x))

        egrid = egrid.view(-1, 169)
        egrid = egrid * 24

        x = torch.cat((x, egrid), dim=1)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sig(self.fc4(x))
        x = x.view(-1, 13, 13)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.salpath = AlexSal()
        self.gazepath = AlexGaze()

    def forward(self, xi, xh, xp):
        outxi = self.salpath(xi)
        outxh = self.gazepath(xh, xp)
        output = outxi * outxh
        return output.view(-1, 169)


model = Net()
xi = Variable(torch.randn(5, 3, 227, 227))
xh = Variable(torch.randn(5, 3, 227, 227))
xp = torch.zeros(5, 13, 13)
xp[0][4][4] = 1

print(model(xi, xh, xp))
