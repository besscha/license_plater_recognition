import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import os

from .models.retina import Retina
from .data import cfg_mnet
from .layers.functions.prior_box import PriorBox
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path='./LP_detect/weights/mobilenet0.25_epoch_20_ccpd.pth'
thres = 0.4

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    #print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model():
    torch.set_grad_enabled(False)
    net=Retina(cfg=cfg_mnet, phase='test')
    weights=torch.load(weights_path,map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in weights.keys():
        weights=remove_prefix(weights['state_dict'],'module.')
    else:
        weights=remove_prefix(weights,'module.')
    net.load_state_dict(weights,strict=False)
    net.eval()
    net=net.to(device)
    return net

def detect_LP(file_path):

    net=load_model()

    # start detecting
    img_raw=cv2.imread(file_path,cv2.IMREAD_COLOR)
    img=np.float32(img_raw)
    im_height,im_width,_=img.shape
    scale=torch.Tensor([img.shape[1],img.shape[0],img.shape[1],img.shape[0]])
    img-=(104,117,123)
    img=img.transpose(2,0,1)
    img=torch.from_numpy(img).unsqueeze(0)
    img=img.to(device)
    scale=scale.to(device)

    loc,conf,landms=net(img) # forward pass

    priorbox=PriorBox(cfg_mnet,image_size=(im_height,im_width))
    priors=priorbox.forward()
    priors=priors.to(device)
    prior_data=priors.data
    boxes=decode(loc.data.squeeze(0),prior_data,cfg_mnet['variance'])
    boxes=boxes*scale
    boxes=boxes.cpu().numpy()

    scores=conf.squeeze(0).data.cpu().numpy()[:,1]

    landms=decode_landm(landms.data.squeeze(0),prior_data,cfg_mnet['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    inds = np.where(scores > 0.02)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:1000]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)
    dets = dets[keep, :]
    landms = landms[keep]

    dets=dets[:500,:]
    landms=landms[:500,:]

    dets = np.concatenate((dets,landms),axis=1)
    for b in dets:
        if b[4] < thres:
            continue
        #print(b)
        b=list(map(int,b))
        w=int(b[2]-b[0]+4.0)
        h=int(b[3]-b[1]+4.0)
        img_box = np.zeros((h,w,3))
        img_box = img_raw[b[1]-2:b[3]+2,b[0]-2:b[2]+2,:]

        new_x1 , new_y1 = b[9] - b[0] , b[10] - b[1]
        new_x2 , new_y2 = b[11] - b[0] , b[12] - b[1]
        new_x3 , new_y3 = b[7] - b[0] , b[8] - b[1]
        new_x4 , new_y4 = b[5] - b[0] , b[6] - b[1]

        points1 = np.float32([[new_x1,new_y1],[new_x2,new_y2],[new_x3,new_y3],[new_x4,new_y4]])
        points2 = np.float32([[0,0],[180,0],[0,40],[180,40]])

        #print(points1)

        M = cv2.getPerspectiveTransform(points1,points2)
        Processed = cv2.warpPerspective(img_box,M,(180,40))
        return Processed