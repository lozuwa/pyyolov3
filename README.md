<h1>Yolov3 module for Python3</h1>
<p>I wanted to be able to use yolov3 as a module so I could use a singleton in a web servie or to embed
it as a microservice. pjreddie's repo has an example that does not work with python3 and is unable to use
images read as numpy arrays. So I made this module which allows you to use yolov3 in python3 by instantiating 
a class. Use the method findObjects to pass an image loaded as a numpy array (BGR format) and get a dictionary of results. Anyway, in case you find this useful, follow these instructions to make it work:</p>

<ol>
  <li>Clone this repo by running git clone https://github.com/lozuwa/pyyolov3.git</li>
  <li>Download yolov3 weights from https://pjreddie.com/darknet/yolo/</li>
  <li>Copy the file yolov3.weights file to the resources folder inside this repo.</li>
  <li>Open the file coco.data inside data/</li>
  <li>Locate the variable names and change its value to the FULL PATH where the file coco.names is located.</li>
  <li>That's it. Use the example script.</li>
</ol>

<h3>Example script</h3>
<p>For the example script you have to make sure that the pyyolov3 folder in in the PYTHONPATH. The easier way to make this work is to export the path inside the repor to PYTHONPATH or just copy pyyolov3 to your site-packages in your python environment so it can be called as a library.</p>

<p>Don't forget to fill the path variable with a valid image.</p>

```python
import cv2
import numpy as np
from pyyolov3.pydarknet import Yolov3

# Read an image.
path:str = "people_1.jpg"
image:np.ndarray = cv2.imread(path)

# Create an instance of yolov3 with no gpu.
yolov3:any = Yolov3(use_gpu=False)

# Run predictions and print them.
objects:dict = yolov3.findObjects(image=image)
print(objects)
```

<h3>GPU usage</h3>
<p>In case you are using your gpu, make sure you have CUDA and CUDNN installed and have the following env variables exported for your user's environment:</p>

export PATH=${PATH}:/usr/local/cuda-9.0/bin
export CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-9.0
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

<p>And just change the use_gpu variable when instantiating Yolov3 to True.</p>

<h3>Further development</h3>
<p>In case anyone wants to automate this installation, he/she is welcome. Thanks.</p>

