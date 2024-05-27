import torch.nn as nn

class CNN_first(nn.Module):
    def __init__(self):
        super(CNN_first,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=5,stride=3,padding=0),   # 46*146
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)   # 23*73
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,kernel_size=5,stride=3,padding=0),   # 7*23
            nn.ReLU()
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=0),   # 5*21
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)   # 2*10
        )
        self.fc=nn.Sequential(
            nn.Linear(1280,640),
            nn.Linear(640,245)
        )

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x