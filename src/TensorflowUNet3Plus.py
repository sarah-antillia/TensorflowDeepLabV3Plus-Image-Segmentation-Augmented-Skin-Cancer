# Copyright 2023 antillia.com Toshiyuki Arai
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

# This is based on the code in the following web sites:

# 1. Keras U-Net starter - LB 0.277
# https://www.kaggle.com/code/keegil/keras-u-net-starter-lb-0-277/notebook

# 2. U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/pdf/1505.04597.pdf

# You can customize your TensorflowUnNet model by using a configration file
# Example: train.config

"""
[model]
image_width    = 256
image_height   = 256
image_channels = 3

num_classes    = 1
base_filters   = 16
num_layers     = 8
dropout_rate   = 0.08
learning_rate  = 0.001
"""

import os
import sys
import datetime

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import sys

import traceback
import random
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D, Dropout, Conv2D, MaxPool2D, BatchNormalization

#from tensorflow.keras.layers import Conv2DTranspose
#from tensorflow.keras.layers import concatenate
#from tensorflow.keras.activations import relu
from tensorflow.keras import Model

from ConfigParser import ConfigParser

import tensorflow as tf
import tensorflow.keras as k
from TensorflowUNet import TensorflowUNet


"""
See: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
Module: tf.keras.metrics
Functions
"""

"""
See also: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py
"""

MODEL  = "model"
TRAIN  = "train"
INFER  = "infer"
# 2023/06/10
TILEDINFER = "tiledinfer"


BEST_MODEL_FILE = "best_model.h5"

class TensorflowUNet3Plus(TensorflowUNet):

  def __init__(self, config_file):
    super().__init__(config_file)


  #The following two methods has been taken from 
  # https://github.com/hamidriasat/UNet-3-Plus/blob/unet3p_lits/models/unet3plus_utils.py
 
  def conv_block(self, x, kernels, kernel_size=(3, 3), strides=(1, 1), padding='same',
               is_bn=True, is_relu=True, n=2):
    """ Custom function for conv2d:
        Apply  3*3 convolutions with BN and relu.
    """
    normalization = False
    for i in range(1, n + 1):
        x = k.layers.Conv2D(filters=kernels, kernel_size=kernel_size,
                            padding=padding, strides=strides,
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                            kernel_initializer=k.initializers.he_normal(seed=5))(x)
        #2023/06/28
        x = Dropout(self.dropout_rate * i)(x)

        if is_bn:
            x = k.layers.BatchNormalization()(x)
        if is_relu:
            x = k.activations.relu(x)

    return x


  def dot_product(self, seg, cls):
    b, h, w, n = k.backend.int_shape(seg)
    seg = tf.reshape(seg, [-1, h * w, n])
    final = tf.einsum("ijk,ik->ijk", seg, cls)
    final = tf.reshape(final, [-1, h, w, n])
    return final
    

  def create(self, num_classes, image_height, image_width, image_channels,
            base_filters = 16, num_layers = 5):
    print("=== TensorflowUNet3Plus  create ...")
    # num_layers is not unsed
    self.dropout_rate = self.config.get(MODEL, "dropout_rate")
    print("=== dropout_rate {}".format(self.dropout_rate))
    output_channels = 1
    #input_shape, output_channels
    input_shape = (image_width, image_height, image_channels)

    num_filters = num_layers
    """ UNet3+ base model """
    #filters = [64, 128, 256, 512, 1024]
    filters = []
    for i in range(num_filters):
      filters.append(base_filters * (2**i))
    print("--- filters {}".format(filters))
   
    input_layer = k.layers.Input(
        shape=input_shape,
        name="input_layer"
    ) 
   
    """ Encoder"""
    # block 1
    e1 = self.conv_block(input_layer, filters[0])

    # block 2
    e2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  
    e2 = self.conv_block(e2, filters[1])

    # block 3
    e3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)
    e3 = self.conv_block(e3, filters[2]) 

    # block 4
    e4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)
    e4 = self.conv_block(e4, filters[3])

    # block 5
    # bottleneck layer
    e5 = k.layers.MaxPool2D(pool_size=(2, 2))(e4) 
    e5 = self.conv_block(e5, filters[4])

    """ Decoder """
    cat_channels = filters[0]
    cat_blocks = len(filters)
    upsample_channels = cat_blocks * cat_channels

    """ d4 """
    e1_d4 = k.layers.MaxPool2D(pool_size=(8, 8))(e1)  
    e1_d4 = self.conv_block(e1_d4, cat_channels, n=1) 

    e2_d4 = k.layers.MaxPool2D(pool_size=(4, 4))(e2)  
    e2_d4 = self.conv_block(e2_d4, cat_channels, n=1) 

    e3_d4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  
    e3_d4 = self.conv_block(e3_d4, cat_channels, n=1) 

    e4_d4 = self.conv_block(e4, cat_channels, n=1)  

    e5_d4 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(e5) 
    e5_d4 = self.conv_block(e5_d4, cat_channels, n=1) 

    d4 = k.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = self.conv_block(d4, upsample_channels, n=1)  

    """ d3 """
    e1_d3 = k.layers.MaxPool2D(pool_size=(4, 4))(e1)  
    e1_d3 = self.conv_block(e1_d3, cat_channels, n=1) 

    e2_d3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  
    e2_d3 = self.conv_block(e2_d3, cat_channels, n=1) 

    e3_d3 = self.conv_block(e3, cat_channels, n=1)  
    
    e4_d3 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d4)  
    e4_d3 = self.conv_block(e4_d3, cat_channels, n=1)  
    
    e5_d3 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(e5) 
    e5_d3 = self.conv_block(e5_d3, cat_channels, n=1)  

    d3 = k.layers.concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3])
    d3 = self.conv_block(d3, upsample_channels, n=1)  

    """ d2 """
    e1_d2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  
    e1_d2 = self.conv_block(e1_d2, cat_channels, n=1) 

    e2_d2 = self.conv_block(e2, cat_channels, n=1)  

    d3_d2 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d3) 
    d3_d2 = self.conv_block(d3_d2, cat_channels, n=1)  

    d4_d2 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d4)  
    d4_d2 = self.conv_block(d4_d2, cat_channels, n=1)  

    e5_d2 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(e5) 
    e5_d2 = self.conv_block(e5_d2, cat_channels, n=1)  

    d2 = k.layers.concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
    d2 = self.conv_block(d2, upsample_channels, n=1)  

    """ d1 """
    e1_d1 = self.conv_block(e1, cat_channels, n=1) 

    d2_d1 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d2)  
    d2_d1 = self.conv_block(d2_d1, cat_channels, n=1)  

    d3_d1 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d3) 
    d3_d1 = self.conv_block(d3_d1, cat_channels, n=1) 

    d4_d1 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(d4) 
    d4_d1 = self.conv_block(d4_d1, cat_channels, n=1) 

    e5_d1 = k.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(e5) 
    e5_d1 = self.conv_block(e5_d1, cat_channels, n=1)  

    d1 = k.layers.concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, ])
    d1 = self.conv_block(d1, upsample_channels, n=1)  

    # last layer does not have batchnorm and relu
    d = self.conv_block(d1, output_channels, n=1, is_bn=False, is_relu=False)

    # 2023/06/29 Modified activate function from softmax to sigmoid 
    #output = k.activations.softmax(d)
    activation = "softmax"
    if num_classes == 1:
      activation = "sigmoid"
    output = tf.keras.layers.Activation(activation=activation)(d)

    return tf.keras.Model(inputs=input_layer, outputs=[output], name='UNet_3Plus')
 
    
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
    model    = TensorflowUNet3Plus(config_file)
    # Please download and install graphviz for your OS
    # https://www.graphviz.org/download/ 
    #image_file = './asset/model.png'
    #tf.keras.utils.plot_model(model.model, to_file=image_file, show_shapes=True)

  except:
    traceback.print_exc()
    
