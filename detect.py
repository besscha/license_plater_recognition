import LP_detect.detect_fix as first_detect
import LP_recongnize.detect as second_detect

import LP_recongnize.Models.CNN as CNN

import cv2
import torch
import torchvision.transforms as transforms

tf=transforms.Compose([transforms.ToTensor()])

if __name__ == '__main__':
    img=first_detect.detect_LP('./testimage/t (2).jpg')
    model=CNN.CNN_MK2()
    t=torch.zeros(1,3,40,180)
    img=tf(img)
    t[0]=img
    output=second_detect.detect(model,'./LP_recongnize/weights/CNN_MK2.pth',t)
    print(output)

