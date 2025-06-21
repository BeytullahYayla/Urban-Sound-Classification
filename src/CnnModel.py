import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        self.layer1 = self.ConvModule(in_channels=3, out_channels=32)
        self.layer2 = self.ConvModule(in_channels=32, out_channels=64)
        self.layer3 = self.ConvModule(in_channels=64, out_channels=128)
        self.layer4 = self.ConvModule(in_channels=128, out_channels=256)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x

    def ConvModule(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

class SE_CustomCNN(CustomCNN):
    def __init__(self, num_classes=10):
        super(SE_CustomCNN, self).__init__(num_classes)
        self.layer1 = self.ConvModule(in_channels=3, out_channels=32)
        self.layer2 = self.ConvModule(in_channels=32, out_channels=64)
        self.layer3 = self.ConvModule(in_channels=64, out_channels=128)
        self.layer4 = self.ConvModule(in_channels=128, out_channels=256)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x
    
    def ConvModule(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
if __name__ == "__main__":
    model = CustomCNN()
    print(model)
    print(model.ConvModule(3,32))
    print(model.ConvModule(32,64))
    print(model.ConvModule(64,128))
    print(model.ConvModule(128,256))
    print(model.classifier)