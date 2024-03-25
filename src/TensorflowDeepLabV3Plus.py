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

# 2024/03/25

# This is based on the code in the following web sites:
# https://github.com/TanyaChutani/DeepLabV3Plus-Tf2.x/blob/master/notebook/DeepLab_V3_Plus.ipynb

# You can customize your TensorflowUnNet model by using a configration file
# Example: train_eval_infer.config


import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import sys
import traceback
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import (Conv2D, Dropout, Conv2D, MaxPool2D, 
                                     Activation, BatchNormalization, UpSampling2D, Concatenate)

from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras import Model
  
from ConfigParser import ConfigParser

from TensorflowUNet import TensorflowUNet

MODEL  = "model"
TRAIN  = "train"
INFER  = "infer"

BEST_MODEL_FILE = "best_model.h5"

# Define TensorflowDeepLabV3Plus class as a subclass of TensorflowUNet

class TensorflowDeepLabV3Plus(TensorflowUNet):

  def __init__(self, config_file):
    super().__init__(config_file)



  def AtrousSpatialPyramidPooling(self, model_input):
    dims = tf.keras.backend.int_shape(model_input)

    layer = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3],
                                                        dims[-2]))(model_input)
    layer = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same',
                                   kernel_initializer = 'he_normal')(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    out_pool = tf.keras.layers.UpSampling2D(size = (dims[-3] // layer.shape[1],
                                                 dims[-2] // layer.shape[2]),
                                          interpolation = 'bilinear')(layer)
    
    layer = tf.keras.layers.Conv2D(256, kernel_size = 1,
                                     dilation_rate = 1, padding = 'same',
                                     kernel_initializer = 'he_normal',
                                     use_bias = False)(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_1 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(256, kernel_size = 3,
                                     dilation_rate = 6, padding = 'same', 
                                     kernel_initializer = 'he_normal',
                                     use_bias = False)(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_6 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(256, kernel_size = 3,
                                     dilation_rate = 12, padding = 'same',
                                     kernel_initializer = 'he_normal',
                                     use_bias = False)(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_12 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(256, kernel_size = 3,
                                     dilation_rate = 18, padding = 'same',
                                     kernel_initializer = 'he_normal',
                                     use_bias = False)(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_18 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Concatenate(axis = -1)([out_pool, out_1,
                                                      out_6, out_12,
                                                      out_18])

    layer = tf.keras.layers.Conv2D(256, kernel_size = 1,
                                     dilation_rate = 1, padding = 'same',
                                     kernel_initializer = 'he_normal',
                                     use_bias = False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    model_output = tf.keras.layers.ReLU()(layer)
    return model_output

       

  #def create(nclasses = 20):
  #  # Customizable by the parameters in a configuration file.
  def create(self, num_classes, image_height, image_width, image_channels,
            base_filters = 16, num_layers = 5):
    # inputs
    print("=== TensorflowMultiResUNet.create ")
    print("Input image_height {} image_width {} image_channels {}".format(image_height, image_width, image_channels))
    inputs = Input((image_height, image_width, image_channels))
    
    resnet50 = tf.keras.applications.ResNet50(weights = 'imagenet',
                                              include_top = False,
                                              input_tensor = inputs)
    layer = resnet50.get_layer('conv4_block6_2_relu').output
    layer = self.AtrousSpatialPyramidPooling(layer)
    input_a = tf.keras.layers.UpSampling2D(size = (image_height // 4 // layer.shape[1],
                                                   image_width // 4 // layer.shape[2]),
                                            interpolation = 'bilinear')(layer)

    input_b = resnet50.get_layer('conv2_block3_2_relu').output
    input_b = tf.keras.layers.Conv2D(48, kernel_size = (1,1), padding = 'same',
                                     kernel_initializer = tf.keras.initializers.he_normal(),
                                     use_bias = False)(input_b)
    input_b = tf.keras.layers.BatchNormalization()(input_b)
    input_b = tf.keras.layers.ReLU()(input_b)

    layer = tf.keras.layers.Concatenate(axis = -1)([input_a, input_b])

    layer = tf.keras.layers.Conv2D(256, kernel_size = 3,
                                     padding = 'same', activation = 'relu',
                                     kernel_initializer = tf.keras.initializers.he_normal(),
                                     use_bias = False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.Conv2D(256, kernel_size =3,
                                     padding = 'same', activation = 'relu',
                                     kernel_initializer = tf.keras.initializers.he_normal(),
                                     use_bias = False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.UpSampling2D(size = (image_height // layer.shape[1],
                                                   image_width // layer.shape[2]),
                                            interpolation = 'bilinear')(layer)
    
    # 2024/03/25 Added the following line
    activation = "softmax"
    if num_classes == 1:
      activation = "sigmoid"
     
    model_output = tf.keras.layers.Conv2D(num_classes, kernel_size = (1,1), activation=activation,
                                     padding = 'same')(layer)
    
    return tf.keras.Model(inputs = inputs, outputs = model_output)
    
    
if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
    print("=== config_file {}".format(config_file))

    config   = ConfigParser(config_file)

    width    = config.get(MODEL, "image_width")
    height   = config.get(MODEL, "image_height")

    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model    = TensorflowDeepLabV3Plus(config_file)
    

  except:
    traceback.print_exc()
    

