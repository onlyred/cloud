import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, in_channel : int, out_channel : int, dropout : float=0.5):
        super(AlexNet, self).__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(in_channel, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, out_channel),
        )
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.cnn(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)  # flatten
        x = self.classifier(x)
        return x
