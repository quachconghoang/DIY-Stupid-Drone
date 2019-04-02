import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from core.inference import get_final_preds
from utils.utils import create_logger

import dataset
import models

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.utils
import cv2 as cv

def displayGT(img_file, joints, center, scale):
    img = cv.imread(img_file)
    img_width = int(192 * scale[0])
    img_height = int(256 * scale[0])
    org_x = int(center[0] - img_width / 2)
    org_y = int(center[1] - img_height / 2)

    # cv.circle(img, (int(center[0]), int(center[1])), 5, (0, 0, 255),2)
    # cv.rectangle(img,(org_x,org_y),(org_x+img_width,org_y+img_height),(0,0,255))

    img = img[org_y:org_y + img_height, org_x:org_x + img_width, :]
    img = cv.resize(img, (192, 256));

    for i in range(joints.shape[0]):
        pt = joints[i]
        cv.circle(img, (int(pt[0]), int(pt[1])), 3, (255, 0, 0))

    cv.imshow('asd', img)
    cv.waitKey()
    cv.destroyAllWindows()

def getCropImg(img_file, center, scale):
    img = cv.imread(img_file)
    img_width = int(192 * scale[0])
    img_height = int(256 * scale[0])
    org_x = int(center[0] - img_width / 2)
    org_y = int(center[1] - img_height / 2)
    img = img[org_y:org_y + img_height, org_x:org_x + img_width, :]
    img = cv.resize(img, (192, 256))
    return img

cfg_path = '/home/hoangqc/Desktop/CppLibraries/SLAM/DIY-Stupid-Drone/Pose/hrnet_w32_256x192.yaml'
cfg.defrost()
cfg.merge_from_file(cfg_path)
cfg.freeze()
# update_config(cfg, cfg_path)

cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


# model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
model = models.pose_hrnet.get_pose_net(cfg, is_train=False)
model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
model.eval()
model.cuda()

# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

valid_dataset = dataset.coco(cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
                             transforms.Compose([transforms.ToTensor(), normalize,]))

batch_data = valid_dataset.__getitem__(101)

input = batch_data[0]
target = batch_data[1]
target_weight = batch_data[2]
meta = batch_data[3]

img_file = meta['image']
joints = meta['joints'] # XY
c = meta['center'] # XY
s = meta['scale'] # 0

img = cv.imread(img_file)


# img = getCropImg(img_file)

# pilImg = Image.open(img_file)
# pilImg.show()

# plt.imshow(input.numpy().transpose(1,2,0));
# plt.show()

img_tensor = input.unsqueeze(0).cuda()

with torch.no_grad():
    for i in range(100):
        e1 = cv.getTickCount()
        out = model(img_tensor)
        e2 = cv.getTickCount()
        print((e2-e1)/cv.getTickFrequency())


# preds, maxvals = get_final_preds(cfg, out.clone().cpu().numpy(), c, s)
#
# for hID in range(preds.shape[0]):
#     for jID in range(preds.shape[1]):
#         pt = preds[hID][jID]
#         cv.circle(img, (int(pt[0]), int(pt[1])), 3, (255, 0, 0))
#
# cv.imshow('asd', img)
# cv.waitKey()
# cv.destroyAllWindows()
