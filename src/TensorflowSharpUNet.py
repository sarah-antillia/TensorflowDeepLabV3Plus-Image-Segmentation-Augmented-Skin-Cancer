# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# 2024/03/20 TensorflowSharpUNet.py

# You can customize your TensorflowUnNet model by using a configration file
# Example: train_eval_infer.config

# 2024/03/23 Modified 'create' method to use for loops to create the encoders and decoders.

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import sys
import traceback
import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import (Conv2D, Dropout, Conv2D, MaxPool2D, MaxPooling2D, DepthwiseConv2D,)

from tensorflow.keras.layers import Conv2DTranspose, AveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras import Model

from ConfigParser import ConfigParser

from TensorflowUNet import TensorflowUNet

MODEL  = "model"
TRAIN  = "train"
INFER  = "infer"

BEST_MODEL_FILE = "best_model.h5"

# The methods in TensorflowSharpUNet class have been taken from
# the following code.
# https://github.com/hasibzunair/sharp-unets/blob/master/demo.ipynb

class TensorflowSharpUNet (TensorflowUNet):

  def __init__(self, config_file):
    super().__init__(config_file)
    print("=== TensorflowSharpUNet.__init__")
    
  def get_kernel(self):
    """
    See https://setosa.io/ev/image-kernels/
    """

    k1 = np.array([[0.0625, 0.125, 0.0625],
                   [0.125,  0.25, 0.125],
                   [0.0625, 0.125, 0.0625]])
    
    # Sharpening Spatial Kernel, used in paper
    k2 = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
    
    k3 = np.array([[0, -1, 0],
                   [-1,  5, -1],
                   [0, -1, 0]])
    
    return k1, k2, k3


  def build_sharp_blocks(self, layer):
    """
    Sharp Blocks
    """
    # Get number of channels in the feature
    in_channels = layer.shape[-1]
    # Get kernel
    _, w, _ = self.get_kernel()    
    # Change dimension
    w = np.expand_dims(w, axis=-1)
    # Repeat filter by in_channels times to get (H, W, in_channels)
    w = np.repeat(w, in_channels, axis=-1)
    # Expand dimension
    w = np.expand_dims(w, axis=-1)
    return w

  def dump(self, convs):
    for i in range(len(convs)):
      print("--- {} dump {}".format(i, convs[i]))

  # 2024/03/23 Modified this method to use for loops to create the encoders and decoders.
  #    Modified to use dilation_rate parameter in Conv2D
  def create(self, num_classes, image_height, image_width, image_channels,
               base_filters = 16, num_layers = 6):
    print("==== TensorflowSharpUNet.create ")
    image_size = (image_width, image_height, image_channels)    

    filters     = self.config.get(MODEL, "filters", dvalue= [32, 64, 128, 256])
    dec_filters = filters[::-1]
    max_filter  = self.config.get(MODEL, "max_filter", dvalue=512)
    print("--- filters     {}".format(filters))
    print("--- dec_filters {}".format(dec_filters))
    print("--- max filter  {}".format(max_filter))
    
    "Unet with sharp Blocks in skip connections"

    base_ksize  = self.config.get(MODEL, "base_ksize", dvalue=(3, 3))
    print("--- base_ksize {}".format(base_ksize))
    dilation = self.config.get(MODEL, "dilation", dvalue=(1, 1))
    print("--- dilation {}".format(dilation))

    # Kernel size for sharp blocks
    kernel_size = 3
    
    inputs = Input(image_size)
    pool = inputs
    enc_convs = []
    layers = len(filters)
    for i in range(layers):
      print("--- {} filter:{}".format(i, filters[i]))
      conv = Conv2D(filters[i], base_ksize, activation='relu', dilation_rate=dilation, padding='same')(pool)
      conv = Conv2D(filters[i], base_ksize, activation='relu', dilation_rate=dilation,padding='same')(conv)
      pool = MaxPooling2D(pool_size=(2, 2))(conv)
      enc_convs = [conv] + enc_convs

    self.dump(enc_convs)
    xconv = Conv2D(max_filter, base_ksize, activation='relu', dilation_rate=dilation, padding='same')(pool)
    xconv = Conv2D(max_filter, base_ksize, activation='relu', dilation_rate=dilation,padding='same')(xconv)

    up = xconv
    print("--- up {}".format(up))
    dec_layers = len(dec_filters)

    for i in range(dec_layers):
      enc = enc_convs[i] 
      print("+++ {} fillter:{} enc:{}".format(i, dec_filters[i], enc))
      # Skip connection 1
      # 1. Get sharpening kernel weights(1, H, W, channels) 
      W1 = self.build_sharp_blocks(enc)
      # 2. Build depthwise convolutional layer with random weights
      sb1 = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
      # 3. Pass input to layer
      conv  = sb1(enc)
      # 4. Set filters as layer weights 
      sb1.set_weights([W1])
      # 5. Dont update weights
      sb1.trainable = False
      up = concatenate([Conv2DTranspose(dec_filters[i], (2, 2), strides=(2, 2), padding='same')(up), conv], axis=3)
      conv = Conv2D(dec_filters[i], base_ksize, activation='relu', dilation_rate=dilation, padding='same')(up)
      conv = Conv2D(dec_filters[i], base_ksize, activation='relu', dilation_rate=dilation, padding='same')(conv)

    # for multi-class segmentation, use the 'softmax' activation
    activation = "softmax" 
    if num_classes == 1:
      activation = "sigmoid"
      
    conv10 = Conv2D(num_classes, (1, 1), activation= activation)(conv)

    model = Model(inputs=[inputs], outputs=[conv10])    
    
    return model

if __name__ == "__main__":
  try:
    # Default config_file
    config_file    = "./train_eval_infer.config"
    # You can specify config_file on your command line parammeter.
    if len(sys.argv) == 2:
      config_file= sys.argv[1]
      if not os.path.exists(config_file):
         raise Exception("Not found " + config_file)
     
    config   = ConfigParser(config_file)
    
    width    = config.get(MODEL, "image_width")
    height   = config.get(MODEL, "image_height")

    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model    = TensorflowSharpUNet(config_file)

  except:
    traceback.print_exc()
