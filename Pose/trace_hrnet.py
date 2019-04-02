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
from utils.utils import create_logger

import dataset
import models

import matplotlib.pyplot as plt
from PIL import Image

cfg_path = '/home/hoangqc/Desktop/CppLibraries/SLAM/DIY-Stupid-Drone/Pose/hrnet_w32_256x192.yaml'
cfg.defrost()
cfg.merge_from_file(cfg_path)
cfg.freeze()
# update_config(cfg, cfg_path)

cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

model = models.pose_hrnet.get_pose_net(cfg, is_train=True)
# model = models.pose_hrnet.PoseHighResolutionNet(cfg)
# model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)

dummy = torch.rand((1, 3,256,192))
# with torch.no_grad():
#     out = model(dummy)

traced_script_module = torch.jit.trace(model, dummy)
output = traced_script_module(torch.ones(1, 1, 256, 192))
traced_script_module.save("hrnet_w32.pt")