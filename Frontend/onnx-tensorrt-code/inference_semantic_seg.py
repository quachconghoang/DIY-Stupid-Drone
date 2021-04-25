import numpy as np
from PIL import Image
import tensorrt as trt
import labels  # from cityscapes evaluation script
import engine as eng
import inference as inf
import keras
import argparse
import skimage.transform

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

MEAN = (71.60167789, 82.09696889, 72.30508881)
CLASSES = 20
HEIGHT = 512
WIDTH = 1024

def sub_mean_chw(data):
   data = data.transpose((1, 2, 0))  # CHW -> HWC
   data -= np.array(MEAN)  # Broadcast subtract
   data = data.transpose((2, 0, 1))  # HWC -> CHW
   return data

def rescale_image(image, output_shape, order=1):
   image = skimage.transform.resize(image, output_shape,
               order=order, preserve_range=True, mode='reflect')
   return image

def color_map(output):
   output = output.reshape(CLASSES, HEIGHT, WIDTH)
   out_col = np.zeros(shape=(HEIGHT, WIDTH), dtype=(np.uint8, 3))
   for x in range(WIDTH):
       for y in range(HEIGHT):

           if (np.argmax(output[:, y, x] )== 19):
               out_col[y,x] = (0, 0, 0)
           else:
               out_col[y, x] = labels.id2label[labels.trainId2label[np.argmax(output[:, y, x])].id].color
   return out_col 


def main(args):

    input_file_path = args.input_image
    serialized_plan_fp32 = args.engine_file
    HEIGHT = args.height
    WIDTH = args.width

    image = np.asarray(Image.open(input_file_path))
    img = rescale_image(image, (HEIGHT, WIDTH),order=1)
    im = np.array(img, dtype=np.float32, order='C')
    im = im.transpose((2, 0, 1))
    im = sub_mean_chw(im)

    engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
    h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)
    out = inf.do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)
    out = color_map(out)

    colorImage_trt = Image.fromarray(out.astype(np.uint8))
    colorImage_trt.save('trt_output.png')

    semantic_model = keras.models.load_model(args.hdf5_file)
    out_keras= semantic_model.predict(im.reshape(-1, 3, HEIGHT, WIDTH))

    out_keras = color_map(out_keras)
    colorImage_k = Image.fromarray(out_keras.astype(np.uint8))
    colorImage_k.save('keras_output.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str)
    parser.add_argument('--engine_file', type=str)
    parser.add_argument('--hdf5_file', type=str)
    parser.add_argument('--height', type=int, default= 512)
    parser.add_argument('--width', type=int, default= 1024)
    args=parser.parse_args()
    main(args)
