# ResNet50
## load ResNet50


You need to install 

segmentation_model:  from https://github.com/qubvel/segmentation_models
keras2onnx:   pip install keras2onnx
tf2onnx: https://github.com/onnx/tensorflow-onnx

Als you need to download the following and put it in the same folder.
cityscapes/labels.py:  https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py



```
python loadResNet.py
```
This will create resnet50.pb
next step is to convert this file to onnx. Go to the directory that you install tf2onnx and then run the following

```
python -m tf2onnx.convert  --input /Path/to/resnet50.pb --inputs input_1:0 --outputs probs/Softmax:0 --output resnet50.onnx
```


## Build Engine:

buildEngine.py creates engine from onnx file. 
It accepts two inputs.
 
--onnx_file : Path to your onnx file

--plan_file : This is the output engine file.

```
python buildEngine.py --onnx_file path/to/resnet50.onnx --plan_file resnet50.plan
```

# Semantic Segmentation
## Load and Save into  ONNX file

You need to download the semantic_segmentation.hdf5 from the link provided in the devblog
```
python loadSemanticSegmentation.py  --hdf5_file path/to/semantic_segmentation.hdf5 
``` 
This will create a semantic_segmentation.onnx file

## Build Engine

```
python buildEngine.py --onnx_file path/to/semantic_segmentation.onnx --plan_file semantic.plan
```
This will create semantic.plan that can will be used later.

## Inference 

Please make sure you download lables.py from Cityscapes github that we provided in the devblog.
Also you need to download a sample image for the input to semantic segmentation model.


```
python inference_semantic_seg.py --input_image path/to/input_image --engine_file path/to/semantic.plan --hdf5_file path/to/semantic_segmentation.hdf5 --height height_of_input --width width_of input
```

This will create two images trt_output.png and keras_output.png. you can compare them. 

# Unet
```
python unet_example.py
