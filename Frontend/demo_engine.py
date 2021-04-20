from engineTRT import *
import tensorrt as trt

model_path_trt = '/home/hoangqc/models/superpoint-320x240.trt'
model_path_onnx = '/home/hoangqc/models/superpoint-320x240.onnx'
shape = [1,1,240,320]


engine = build_engine(model_path_onnx, shape=[1,1,240,320])
save_engine(engine,'test.trt')