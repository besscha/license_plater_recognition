import json
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

class train_data(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.file_path='./LP_recongnize/dataset/train_data/'
        with open('./LP_recongnize/dataset/label_train.json','r') as f:
            self.name_dict=json.load(f)

    def __len__(self):
        return len(self.name_dict)
    
    def __getitem__(self, index):
        LP=self.name_dict[index]['name']
        l=self.name_dict[index]['label']
        label=torch.zeros(245)
        for i in range(7):
            label[int(l[i])+35*i]=1.0
        img=Image.open(self.file_path+LP)
        img=tf(img)
        return img, label
    
class test_data(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.file_path='./LP_recongnize/dataset/test_data/'
        with open('./LP_recongnize/dataset/label_test.json','r') as f:
            self.name_dict=json.load(f)

    def __len__(self):
        return len(self.name_dict)
    
    def __getitem__(self, index):
        LP=self.name_dict[index]['name']
        l=self.name_dict[index]['label']
        label=torch.zeros(245)
        for i in range(7):
            label[int(l[i])+35*i]=1.0
        img=Image.open(self.file_path+LP)
        img=tf(img)
        return img, label