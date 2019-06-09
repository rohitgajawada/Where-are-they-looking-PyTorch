from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import cv2
import argparse
import numpy as np
import net_s3fd
from skimage import io, transform
from bbox import *
import matplotlib.pyplot as plt
import torchvision

def detect(net,img):
    img = img - np.array([104,117,123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,)+img.shape)

    img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
    BB,CC,HH,WW = img.size()
    olist = net(img)

    bboxlist = []
    for i in range(len(olist)/2): olist[i*2] = F.softmax(olist[i*2])
    for i in range(len(olist)/2):
        ocls,oreg = olist[i*2].data.cpu(),olist[i*2+1].data.cpu()
        FB,FC,FH,FW = ocls.size() # feature map size
        stride = 2**(i+2)    # 4,8,16,32,64,128
        anchor = stride*4
        for Findex in range(FH*FW):
            windex,hindex = Findex%FW,Findex//FW
            axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
            score = ocls[0,1,hindex,windex]
            loc = oreg[0,:,hindex,windex].contiguous().view(1,4)
            if score<0.05: continue
            priors = torch.Tensor([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]])
            variances = [0.1,0.2]
            box = decode(loc,priors,variances)
            x1,y1,x2,y2 = box[0]*1.0

            bboxlist.append([x1,y1,x2,y2,score])
    bboxlist = np.array(bboxlist)
    if 0==len(bboxlist): bboxlist=np.zeros((1, 5))
    return bboxlist

def getFaces(img):

    parser = argparse.ArgumentParser(description='PyTorch face detect')
    parser.add_argument('--net','-n', default='s3fd', type=str)
    parser.add_argument('--model', default='../s3fd_convert.pth', type=str)


    args = parser.parse_args()
    net = getattr(net_s3fd,args.net)()
    if args.model!='' :net.load_state_dict(torch.load(args.model))
    else: print('Please set --model parameter!')
    net.cuda()
    net.eval()

    shp = img.shape
    bboxlist = detect(net,img)
    keep = nms(bboxlist,0.3)
    bboxlist = bboxlist[keep,:]

    faces = torch.FloatTensor()
    eyecoords = []
    eye_grid = torch.FloatTensor()


    for b in bboxlist:
        x1,y1,x2,y2,s = b
        if s<0.5: continue

        face = img[int(y1):int(y2), int(x1):int(x2)]
        try:
            face = np.ascontiguousarray(face)
            face = transform.resize(face,(227, 227))
            face = face.transpose((2, 0, 1))
        except:
            continue

        face = torch.from_numpy(face).contiguous()
        face = face.view(1, 3, 227, 227)
        face = face.float()

        faces = torch.cat((faces, face))
        ey_x = (x1 + x2)/2
        ey_y = (y1 + y2)/2
        print(ey_x, ey_y, shp[1], shp[0])
        print("Face: ", x1, x2, y1, y2)
        ey_x = int(np.floor((ey_x/float(shp[1]))*13.0))
        ey_y = int(np.floor((ey_y/float(shp[0]))*13.0))
        if ey_x >= 12:
            ey_x = 12
        if ey_y >= 12:
            ey_y = 12
        eye = np.zeros((13,13))
        eye[ey_y][ey_x] = 1
        eye = torch.from_numpy(eye)
        eye = eye.view(1, 1, 13, 13)
        eye_grid = torch.cat((eye_grid, eye.float()))
        ix = (x1 + x2)/(2.0 * float(shp[1]))
        iy = (y1 + y2)/(2.0 * float(shp[0]))
        eyecoords.append((ix, iy))

    return faces, eye_grid, eyecoords
