import torch
import torch.onnx

from demo_superpoint import SuperPointFrontend, SuperPointNet

# WIDTH = 320; HEIGHT = 240
WIDTH = 640; HEIGHT = 480
# WIDTH = 376; HEIGHT = 240
# WIDTH = 752; HEIGHT = 480

model = SuperPointNet()
state_dict = torch.load('superpoint_v1.pth')
model.load_state_dict(state_dict)

# Create the right input shape (e.g. for an image)
dummy_input = torch.randn(1, 1, HEIGHT, WIDTH)
# torch.onnx.export(model, dummy_input, "superpoint-320x240.onnx")

traced_script_module = torch.jit.trace(model, dummy_input)
output = traced_script_module(torch.ones(1, 1, HEIGHT, WIDTH))

# traced_script_module.save("superpoint_v1_320x240.pt")
traced_script_module.save("superpoint_v1_640x480.pt")
# traced_script_module.save("superpoint_v1_376x240.pt")
# traced_script_module.save("superpoint_v1_752x480.pt")
