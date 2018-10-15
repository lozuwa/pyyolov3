import os
import math
import random
from ctypes import *
import numpy as np

# Path to graphs.
dir_path = os.path.dirname(os.path.realpath(__file__))

class BOX(Structure):
  _fields_ = [("x", c_float),
              ("y", c_float),
              ("w", c_float),
              ("h", c_float)]

class DETECTION(Structure):
  _fields_ = [("bbox", BOX),
              ("classes", c_int),
              ("prob", POINTER(c_float)),
              ("mask", POINTER(c_float)),
              ("objectness", c_float),
              ("sort_class", c_int)]

class IMAGE(Structure):
  _fields_ = [("w", c_int),
              ("h", c_int),
              ("c", c_int),
              ("data", POINTER(c_float))]

class METADATA(Structure):
  _fields_ = [("classes", c_int),
              ("names", POINTER(c_char_p))]

class Yolov3Wrapper(object):
  def __init__(self, use_gpu:bool=None):
    super(Yolov3Wrapper, self).__init__()
    # Assertions.
    if use_gpu==None:
      use_gpu=False
    # Local variables.
    if use_gpu:
      path_to_static_lib = os.path.join(dir_path, "resources/libdarknet_gpu.so")
    else:
      path_to_static_lib = os.path.join(dir_path, "resources/libdarknet_cpu.so")
    # Load resources.
    lib = CDLL(path_to_static_lib, RTLD_GLOBAL)
    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int

    predict = lib.network_predict
    predict.argtypes = [c_void_p, POINTER(c_float)]
    predict.restype = POINTER(c_float)

    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

    make_image = lib.make_image
    make_image.argtypes = [c_int, c_int, c_int]
    make_image.restype = IMAGE

    get_network_boxes = lib.get_network_boxes
    get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
    get_network_boxes.restype = POINTER(DETECTION)

    # Class variable.
    self.get_network_boxes = get_network_boxes

    make_network_boxes = lib.make_network_boxes
    make_network_boxes.argtypes = [c_void_p]
    make_network_boxes.restype = POINTER(DETECTION)

    free_detections = lib.free_detections
    free_detections.argtypes = [POINTER(DETECTION), c_int]

    # Class variable.
    self.free_detections = free_detections

    free_ptrs = lib.free_ptrs
    free_ptrs.argtypes = [POINTER(c_void_p), c_int]

    network_predict = lib.network_predict
    network_predict.argtypes = [c_void_p, POINTER(c_float)]

    reset_rnn = lib.reset_rnn
    reset_rnn.argtypes = [c_void_p]

    load_net = lib.load_network
    load_net.argtypes = [c_char_p, c_char_p, c_int]
    load_net.restype = c_void_p

    # Class variable.
    self.load_net = load_net

    do_nms_obj = lib.do_nms_obj
    do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    # Class variable.
    self.do_nms_obj = do_nms_obj

    do_nms_sort = lib.do_nms_sort
    do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    free_image = lib.free_image
    free_image.argtypes = [IMAGE]

    letterbox_image = lib.letterbox_image
    letterbox_image.argtypes = [IMAGE, c_int, c_int]
    letterbox_image.restype = IMAGE

    load_meta = lib.get_metadata
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA

    # Class variable.
    self.load_meta = load_meta

    load_image = lib.load_image_color
    load_image.argtypes = [c_char_p, c_int, c_int]
    load_image.restype = IMAGE

    rgbgr_image = lib.rgbgr_image
    rgbgr_image.argtypes = [IMAGE]

    # Class variable.
    self.rgbgr_image = rgbgr_image

    predict_image = lib.network_predict_image
    predict_image.argtypes = [c_void_p, IMAGE]
    predict_image.restype = POINTER(c_float)

    # Class variable.
    self.predict_image = predict_image

  def c_array(self, ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class Yolov3(Yolov3Wrapper):
  def __init__(self, use_gpu:bool=None):
    super(Yolov3, self).__init__(use_gpu=use_gpu)
    path_to_yolo_v3_cfg = os.path.join(dir_path, "resources/yolov3.cfg").encode("ascii")
    path_to_yolo_v3_weights = os.path.join(dir_path, "resources/yolov3.weights").encode("ascii")
    path_to_coco_data = os.path.join(dir_path, "data/coco.data").encode("ascii")
    self.net = self.load_net(path_to_yolo_v3_cfg, path_to_yolo_v3_weights, 0)
    self.meta = self.load_meta(path_to_coco_data)

  def arrayToImage(self, arr):
    arr = arr.transpose(2,0,1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

  def findObjects(self, image:np.ndarray=None, thresh:float=None, hier_thresh:float=None, nms:float=None):
    """
    Infers the objects in an image.
    :param image: A numpy array that contains an image. 
    :param thresh: The detection threshold.
    :param hier_thresh: The hierarchical threshold.
    :param nms: non-max-supression parameter.
    """
    # Assertions.
    if not isinstance(image, np.ndarray):
      raise ValueError("Image cannot be None.")
    if not isinstance(thresh, float):
      thresh = 0.5
    if not isinstance(hier_thresh, float):
      hier_thresh = 0.5
    if not isinstance(nms, float):
      nms = 0.45
    # Logic.
    im, image = self.arrayToImage(image)
    self.rgbgr_image(im)
    num = c_int(0)
    pnum = pointer(num)
    self.predict_image(self.net, im)
    dets = self.get_network_boxes(self.net, im.w, im.h, thresh, 
                            hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms:
      self.do_nms_obj(dets, num, self.meta.classes, nms)
    res = {}
    for j in range(num):
      a = dets[j].prob[0:self.meta.classes]
      if any(a):
        ai = np.array(a).nonzero()[0]
        for i in ai:
          b = dets[j].bbox
          height = int(b.h)//2
          width = int(b.w)//2
          top, right, bottom, left = int(b.y)-height, int(b.x)+width, int(b.y)+height,int(b.x)-width
          if self.meta.names[i] not in res.keys():
            res[self.meta.names[i]] = []
          res[self.meta.names[i]].append({'score': dets[j].prob[i],
                      'bbox': [top, right, bottom, left]})
    self.free_detections(dets, num)
    return res

