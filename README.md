In case you find this useful, follow these instructions to make it work:

<ol>
  <li>Download yolov3 weights from https://pjreddie.com/darknet/yolo/</li>
  <li>Copy the yolov3.weights file to the resources folder.</li>
  <li>Run pydarknet_test.py to</li>
</ol>

<h3>GPU usage</h3>
In case you are using your gpu, make sure you have the following env variables exported in your environment:

export PATH=${PATH}:/usr/local/cuda-9.0/bin
export CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-9.0
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64






