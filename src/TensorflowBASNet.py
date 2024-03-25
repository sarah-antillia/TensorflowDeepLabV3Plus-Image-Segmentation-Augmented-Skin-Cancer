
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

# This is based on the code in the following web sites:

# https://keras.io/examples/vision/basnet_segmentation/


# You can customize your TensorflowUnNet model by using a configration file
# Example: train_eval_infer.config


import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import shutil
import sys
import glob
import traceback
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import (Add, Conv2D, Dropout, Conv2D, MaxPool2D, Resizing, concatenate,
                                     Activation, BatchNormalization, UpSampling2D, Concatenate)

from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.activations import elu, relu
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import keras_cv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend
from ConfigParser import ConfigParser

from TensorflowUNet import TensorflowUNet

MODEL  = "model"
TRAIN  = "train"
INFER  = "infer"

BEST_MODEL_FILE = "best_model.h5"

class TensorflowBASNet(TensorflowUNet):

  def __init__(self, config_file):
    super().__init__(config_file)


  def basic_block(self, x_input, filters, stride=1, down_sample=None, activation=None):
    """Creates a residual(identity) block with two 3*3 convolutions."""
    residual = x_input

    x = Conv2D(filters, (3, 3), strides=stride, padding="same", use_bias=False)(
        x_input
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, (3, 3), strides=(1, 1), padding="same", use_bias=False)(
        x
    )
    x = BatchNormalization()(x)

    if down_sample is not None:
        residual = down_sample

    x = layers.Add()([x, residual])

    if activation is not None:
        x = Activation(activation)(x)

    return x


  def convolution_block(self, x_input, filters, dilation=1):
    """Apply convolution + batch normalization + relu layer."""
    x = Conv2D(filters, (3, 3), padding="same", dilation_rate=dilation)(x_input)
    x = BatchNormalization()(x)
    return Activation("relu")(x)


  def segmentation_head(self, x_input, out_classes, final_size):
    """Map each decoder stage output to model output classes."""
    x = Conv2D(out_classes, kernel_size=(3, 3), padding="same")(x_input)

    if final_size is not None:
        x = Resizing(final_size[0], final_size[1])(x)

    return x


  def get_resnet_block(self, _resnet, block_num):
    """Extract and return ResNet-34 block."""
    resnet_layers = [3, 4, 6, 3]  # ResNet-34 layer sizes at different block.
    return keras.models.Model(
        inputs=_resnet.get_layer(f"v2_stack_{block_num}_block1_1_conv").input,
        outputs=_resnet.get_layer(
            f"v2_stack_{block_num}_block{resnet_layers[block_num]}_add"
        ).output,
        name=f"resnet34_block{block_num + 1}",
    )
  
  def basnet_predict(self, input_shape, out_classes):
    """BASNet Prediction Module, it outputs coarse label map."""
    filters = 64
    num_stages = 6

    x_input = Input(input_shape)

    # -------------Encoder--------------
    x = Conv2D(filters, kernel_size=(3, 3), padding="same")(x_input)

    resnet = keras_cv.models.ResNet34Backbone(
        include_rescaling=False,
    )

    encoder_blocks = []
    for i in range(num_stages):
        if i < 4:  # First four stages are adopted from ResNet-34 blocks.
            x = self.get_resnet_block(resnet, i)(x)
            encoder_blocks.append(x)
            x = Activation("relu")(x)
        else:  # Last 2 stages consist of three basic resnet blocks.
            x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
            x = self.basic_block(x, filters=filters * 8, activation="relu")
            x = self.basic_block(x, filters=filters * 8, activation="relu")
            x = self.basic_block(x, filters=filters * 8, activation="relu")
            encoder_blocks.append(x)

    # -------------Bridge-------------
    x = self.convolution_block(x, filters=filters * 8, dilation=2)
    x = self.convolution_block(x, filters=filters * 8, dilation=2)
    x = self.convolution_block(x, filters=filters * 8, dilation=2)
    encoder_blocks.append(x)

    # -------------Decoder-------------
    decoder_blocks = []
    for i in reversed(range(num_stages)):
        if i != (num_stages - 1):  # Except first, scale other decoder stages.
            shape = keras.backend.int_shape(x)
            x = Resizing(shape[1] * 2, shape[2] * 2)(x)

        x = concatenate([encoder_blocks[i], x], axis=-1)
        x = self.convolution_block(x, filters=filters * 8)
        x = self.convolution_block(x, filters=filters * 8)
        x = self.convolution_block(x, filters=filters * 8)
        decoder_blocks.append(x)

    decoder_blocks.reverse()  # Change order from last to first decoder stage.
    decoder_blocks.append(encoder_blocks[-1])  # Copy bridge to decoder.

    # -------------Side Outputs--------------
    decoder_blocks = [
        self.segmentation_head(decoder_block, out_classes, input_shape[:2])
        for decoder_block in decoder_blocks
    ]

    return Model(inputs=[x_input], outputs=decoder_blocks)

  def basnet_rrm(self, base_model, out_classes):
    """BASNet Residual Refinement Module(RRM) module, output fine label map."""
    num_stages = 4
    filters = 64

    x_input = base_model.output[0]

    # -------------Encoder--------------
    x = Conv2D(filters, kernel_size=(3, 3), padding="same")(x_input)

    encoder_blocks = []
    for _ in range(num_stages):
        x = self.convolution_block(x, filters=filters)
        encoder_blocks.append(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # -------------Bridge--------------
    x = self.convolution_block(x, filters=filters)

    # -------------Decoder--------------
    for i in reversed(range(num_stages)):
        shape = keras.backend.int_shape(x)
        x = Resizing(shape[1] * 2, shape[2] * 2)(x)
        x = concatenate([encoder_blocks[i], x], axis=-1)
        x = self.convolution_block(x, filters=filters)

    x = self.segmentation_head(x, out_classes, None)  # Segmentation head.

    # ------------- refined = coarse + residual
    x = Add()([x_input, x])  # Add prediction + refinement output

    return Model(inputs=[base_model.input], outputs=[x])

  def create(self, num_classes, image_height, image_width, image_channels,
               base_filters = 16, num_layers = 6):
    print("==== TensorflowBASNet.create ")
    image_size = (image_width, image_height, image_channels)
    out_classes = num_classes  

    """BASNet, it's a combination of two modules
    Prediction Module and Residual Refinement Module(RRM)."""
    input_shape =  (image_width, image_height, image_channels)
    # Prediction model.
    predict_model = self.basnet_predict(input_shape, out_classes)
    # Refinement model.
    refine_model =  self.basnet_rrm(predict_model, out_classes)

    output = [refine_model.output]  # Combine outputs.
    output.extend(predict_model.output)

    output = [Activation("sigmoid")(_) for _ in output]  # Activations.

    return Model(inputs=[predict_model.input], outputs=output)

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
    model    = TensorflowBASNet(config_file)
    
  except:
    traceback.print_exc()
    
