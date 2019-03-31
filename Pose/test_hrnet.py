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


# model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
model = models.pose_hrnet.get_pose_net(cfg, is_train=False)
model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
model.eval()
model.cuda()

# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

valid_dataset = dataset.coco(cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
                             transforms.Compose([transforms.ToTensor(), normalize,]))

img_file = valid_dataset.__getitem__(0)[3]['image']
img = Image.open(img_file)
img.show()

data = valid_dataset.__getitem__(0)
img_tensor = valid_dataset.__getitem__(0)[0].unsqueeze(0).cuda()

# valid_loader = torch.utils.data.DataLoader(
#     valid_dataset,
#     batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
#     shuffle=False,
#     num_workers=cfg.WORKERS,
#     pin_memory=True
# )

# for i, (input, target, target_weight, meta) in enumerate(valid_loader):
#     print(input.shape)

with torch.no_grad():
    out = model(img_tensor)


