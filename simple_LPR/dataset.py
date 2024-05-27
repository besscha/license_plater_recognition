import json
import os
import PIL.Image as Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

lp_chinese_dict = {'京':1,'沪':2,'津':3,'渝':4,'冀':5,'晋':6,'蒙':7,'辽':8,'吉':9,'黑':10,'苏':11,'浙':12,'皖':13,'闽':14,'赣':15,'鲁':16,'豫':17,'鄂':18,'湘':19,'粤':20,'桂':21,'琼':22,'川':23,'贵':24,'云':25,'藏':26,'陕':27,'甘':28,'青':29,'宁':30,'新':31}
lp_other_dict = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'J':9,'K':10,'L':11,'M':12,'N':13,'P':14,'Q':15,'R':16,'S':17,'T':18,'U':19,'V':20,'W':21,'X':22,'Y':23,'Z':24,'0':25,'1':26,'2':27,'3':28,'4':29,'5':30,'6':31,'7':32,'8':33,'9':34,'-':35}
tf=transforms.Compose([transforms.ToTensor()])

class train_data(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.file_path='simple_LPR/dataset/train_data/'
        with open(self.file_path+'name_dict.json','r') as f:
            self.name_dict=json.load(f)
        
    def __len__(self):
        return len(self.name_dict)
    
    def __getitem__(self, index):
        LP=self.name_dict[index]['name']
        l=[]
        l.append(lp_chinese_dict[LP[0]])
        for i in range(1,7):
            l.append(lp_other_dict[LP[i]])
        label=torch.zeros(245)
        for i in range(7):
            label[int(l[i]+35*i)]=1.0
        img=Image.open(self.file_path+str(index)+'.jpg')
        img=tf(img)
        return img, label

class test_data(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.file_path='simple_LPR/dataset/test_data/'
        with open(self.file_path+'name_dict.json','r') as f:
            self.name_dict=json.load(f)
        
    def __len__(self):
        return len(self.name_dict)
    
    def __getitem__(self, index):
        LP=self.name_dict[index]['name']
        l=[]
        l.append(lp_chinese_dict[LP[0]])
        for i in range(1,7):
            l.append(lp_other_dict[LP[i]])
        label=torch.zeros(245)
        for i in range(7):
            label[l[i]+35*i]=1.0
        img=Image.open(self.file_path+str(index)+'.jpg')
        img=tf(img)
        return img, label

if __name__=='__main__':
    with open('simple_LPR/dataset/train_data/name_dict.json','r') as f:
        name_dict=json.load(f)
    i=name_dict[0]['name']
    print(i.shape)
    n=np.array(i)
    print(n.shape)