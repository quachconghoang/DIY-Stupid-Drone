import keras
from keras2onnx import convert_keras
import argparse


def keras_to_onnx(model, output_filename):
   onnx = convert_keras(model, output_filename)
   with open(output_filename, "wb") as f:
       f.write(onnx.SerializeToString())


def main(args):
    semantic_model = keras.models.load_model(args.hdf5_file)
    keras_to_onnx(semantic_model, args.onnx_file) 
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_file', type=str)
    parser.add_argument('--onnx_file', type=str, default='semantic_segmentation.onnx')
    args=parser.parse_args()
    main(args)