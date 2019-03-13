import cv2
import torch

import numpy as np
import h5py

from demo_superpoint import SuperPointFrontend, SuperPointNet
import matplotlib.pyplot as plt

net = SuperPointNet()
net.load_state_dict(torch.load('superpoint_v1.pth'))
net.cuda()

W = 160
H = 120

img_path = './assets/icl_snippet/250.png'
img_gray = cv2.imread(img_path,0)
img_gray = cv2.resize(img_gray, (W, H),interpolation=cv2.INTER_AREA)
img_gray = (img_gray.astype('float32') / 255.)

# cv2.imshow('test',img_gray);cv2.waitKey();cv2.destroyAllWindows()

inp = img_gray.copy()
inp = (inp.reshape(1, H, W))
inp = torch.from_numpy(inp)
inp = torch.autograd.Variable(inp).view(1, 1, H, W)

inp = inp.cuda()
outs = net.forward(inp)
semi, coarse_desc = outs[0], outs[1]

semi = semi.data.cpu().numpy().squeeze()
coarse_desc = coarse_desc.data.cpu().numpy().squeeze()

# h5f = h5py.File('rs.h5', 'w')
# h5f.create_dataset('semi', data=semi)
# h5f.create_dataset('coarse_desc', data=coarse_desc)
# h5f.close()

dense = np.exp(semi)  # Softmax.
dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
nodust = dense[:-1, :, :]

Hc = int(H/8)
Wc = int(W/8)
nodust = nodust.transpose(1, 2, 0)
heatmap = np.reshape(nodust, [Hc, Wc, 8, 8])
heatmap = np.transpose(heatmap, [0, 2, 1, 3])
heatmap = np.reshape(heatmap, [Hc * 8, Wc * 8])

# dummy_input = torch.randn(1, 1, H, W).cuda()
# traced_script_module = torch.jit.trace(net, dummy_input)
# output = traced_script_module(torch.ones(1, 1, H, W).cuda())
# traced_script_module.save("superpoint_v1.pt")