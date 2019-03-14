import cv2
import torch

import numpy as np

from demo_superpoint import SuperPointFrontend, SuperPointNet
import matplotlib.pyplot as plt


def nms_fast(self, in_corners, H, W, dist_thresh):

    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

net = SuperPointNet()
net.load_state_dict(torch.load('superpoint_v1.pth'))
net.cuda()

W = 160
H = 120
cell = 8

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

dense = np.exp(semi)  # Softmax.
dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
nodust = dense[:-1, :, :]

Hc = int(H/cell)
Wc = int(W/cell)
nodust = nodust.transpose(1, 2, 0)
heatmap = np.reshape(nodust, [Hc, Wc, cell, cell])
heatmap = np.transpose(heatmap, [0, 2, 1, 3])
heatmap = np.reshape(heatmap, [Hc * cell, Wc * cell])

conf_thresh = 0.015
xs, ys = np.where(heatmap >= conf_thresh) # Confidence threshold.
pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
pts[0, :] = ys
pts[1, :] = xs
pts[2, :] = heatmap[xs, ys]

# dummy_input = torch.randn(1, 1, H, W).cuda()
# traced_script_module = torch.jit.trace(net, dummy_input)
# output = traced_script_module(torch.ones(1, 1, H, W).cuda())
# traced_script_module.save("superpoint_v1.pt")