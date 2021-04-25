import segmentation_models as sm
import keras
from keras2onnx import convert_keras
from engine import *
 
onnx_path = 'unet.onnx'
engine_name = 'unet.plan'
batch_size = 1
CHANNEL = 3
HEIGHT = 224
WIDTH = 224
 
 
model = sm.Unet()
model._layers[0].batch_input_shape = (None, 224,224,3)
model = keras.models.clone_model(model)
 
onx = convert_keras(model, onnx_path)
with open(onnx_path, "wb") as f:
    f.write(onx.SerializeToString())
 

shape = [batch_size , HEIGHT, WIDTH, CHANNEL]
engine = build_engine(onnx_path, shape= shape)
save_engine(engine, engine_name) 