import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,128,kernel_size=5,stride=5,padding=0),   # 8*36
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)   # 4*18
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(128,256,kernel_size=2,stride=1,padding=0),   # 4*18
            nn.ReLU()
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(256,128,kernel_size=2,stride=2,padding=1),   # 2*9
            nn.ReLU(),
        )
        self.fc=nn.Sequential(
            nn.Linear(2304,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,648),
            nn.ReLU(),
            nn.Linear(648,245)
        )
    
    def forward(self,x):
        x=self.conv1(x)
        #print(x.size())
        x=self.conv2(x)
        #print(x.size())
        x=self.conv3(x)
        #print(x.size())
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x
    
class CNN_MK1(nn.Module):
    def __init__(self):
        super(CNN_MK1,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,128,kernel_size=3,stride=1,padding=1),  # 40*180
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)   # 20*90
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),   # 20*90
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(256,360,kernel_size=3,stride=1,padding=1),   # 20*90
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)   # 10*45
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(360,256,kernel_size=3,stride=1,padding=1),   # 10*45
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(256,126,kernel_size=3,stride=1,padding=1),   # 10*45
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)   # 5*22
        )
        '''self.conv6=nn.Sequential(
            nn.Conv2d(126,64,kernel_size=3,stride=1,padding=1),   # 5*22
            nn.ReLU(),
        )
        self.fc1=nn.Sequential( 
                nn.Linear(7040,3080),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        self.fc2=nn.Sequential(
                nn.Linear(3080,245),
                nn.ReLU()
            )'''
        self.fc = nn.ModuleList()

        for i in range(7):
            fc1 = nn.Sequential(
                nn.Linear(1980,990),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            fc2 = nn.Sequential(
                nn.Linear(990,35),
                nn.ReLU()
            )
            self.fc.append(nn.Sequential(fc1,fc2))

                

    
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        #print(x.size())
        x=torch.chunk(x,7,dim=1)
        x=[i for i in x]
        for i in range(7):
            x[i]=x[i].view(x[i].size(0),-1)
            x[i]=self.fc[i](x[i])
        x=torch.cat(x,dim=1)
        return x

class CNN_MK2(nn.Module):
    def __init__(self):
        super(CNN_MK2,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,128,kernel_size=3,stride=1,padding=1),  # 40*180
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)   # 20*90
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),   # 20*90
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(256,360,kernel_size=3,stride=1,padding=1),   # 20*90
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)   # 10*45
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(360,256,kernel_size=3,stride=1,padding=1),   # 10*45
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(256,126,kernel_size=3,stride=1,padding=1),   # 10*45
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)   # 5*22
        )
        self.conv6=nn.Sequential(
            nn.Conv2d(126,64,kernel_size=3,stride=1,padding=1),   # 5*22
            nn.ReLU(),
        )
        self.fc1=nn.Sequential( 
                nn.Linear(7040,3080),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        self.fc2=nn.Sequential(
                nn.Linear(3080,245),
                nn.ReLU()
            )

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.conv6(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.fc2(x)
        return x
       
