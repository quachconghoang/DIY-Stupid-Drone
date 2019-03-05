import torch
import torch.onnx

from demo_superpoint import SuperPointFrontend, SuperPointNet

model = SuperPointNet()
state_dict = torch.load('superpoint_v1.pth')
model.load_state_dict(state_dict)

# Create the right input shape (e.g. for an image)
dummy_input = torch.randn(1, 1, 320, 320)
torch.onnx.export(model, dummy_input, "superpoint-320x320.onnx")