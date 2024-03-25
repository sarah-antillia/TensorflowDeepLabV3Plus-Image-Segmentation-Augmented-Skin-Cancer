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

# TensorflowTransUNet.py
# 2024/03/07 to-arai

"""
This is based on the following code in 
# keras-unet-collection
# https://github.com/yingkaisha/keras-unet-collection/tree/main/keras_unet_collection

"""


import os
import sys
import traceback
import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, concatenate
from keras_unet_collection.layer_utils import *
#from keras_unet_collection.activations import GELU, Snake
from keras_unet_collection._model_unet_2d import UNET_left, UNET_right
from keras_unet_collection.transformer_layers import patch_extract, patch_embedding
from keras_unet_collection._backbone_zoo import backbone_zoo, bach_norm_checker

from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization, Dense, Embedding
    
import sys

# keras-unet-collection
# https://github.com/yingkaisha/keras-unet-collection/tree/main/keras_unet_collection

from TensorflowUNet import TensorflowUNet

TRAIN = "train"
MODEL = "model"
EVAL  = "eval"
INFER = "infer"

class TensorflowTransUNet(TensorflowUNet) :
  
  # 2024/03/25 Modified to call super()
  def __init__(self, config_file):
    super().__init__(config_file)
    print("=== TensorflowTransUNet.__init__ {}".format(config_file))

  
  def ViT_MLP(self, X, filter_num, activation='GELU', name='MLP'):
      '''
      The MLP block of ViT.
      
      ----------
      Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, 
      T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. 
      An image is worth 16x16 words: Transformers for image recognition at scale. 
      arXiv preprint arXiv:2010.11929.
      
      Input
      ----------
          X: the input tensor of MLP, i.e., after MSA and skip connections
          filter_num: a list that defines the number of nodes for each MLP layer.
                          For the last MLP layer, its number of node must equal to the dimension of key.
          activation: activation of MLP nodes.
          name: prefix of the created keras layers.
          
      Output
      ----------
          V: output tensor.

      '''
      activation_func = eval(activation)
      
      for i, f in enumerate(filter_num):
          X = Dense(f, name='{}_dense_{}'.format(name, i))(X)
          X = activation_func(name='{}_activation_{}'.format(name, i))(X)
          
      return X
      
  def ViT_block(self, V, num_heads, key_dim, filter_num_MLP, activation='GELU', name='ViT'):
      '''
      
      Vision transformer (ViT) block.
      
      ViT_block(V, num_heads, key_dim, filter_num_MLP, activation='GELU', name='ViT')
      
      ----------
      Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, 
      T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. 
      An image is worth 16x16 words: Transformers for image recognition at scale. 
      arXiv preprint arXiv:2010.11929.
      
      Input
      ----------
          V: embedded input features.
          num_heads: number of attention heads.
          key_dim: dimension of the attention key (equals to the embeded dimensions).
          filter_num_MLP: a list that defines the number of nodes for each MLP layer.
                          For the last MLP layer, its number of node must equal to the dimension of key.
          activation: activation of MLP nodes.
          name: prefix of the created keras layers.
          
      Output
      ----------
          V: output tensor.
      
      '''
      # Multiheaded self-attention (MSA)
      V_atten = V # <--- skip
      V_atten = LayerNormalization(name='{}_layer_norm_1'.format(name))(V_atten)
      V_atten = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, 
                                   name='{}_atten'.format(name))(V_atten, V_atten)
      # Skip connection
      V_add = add([V_atten, V], name='{}_skip_1'.format(name)) # <--- skip
      
      # MLP
      V_MLP = V_add # <--- skip
      V_MLP = LayerNormalization(name='{}_layer_norm_2'.format(name))(V_MLP)
      V_MLP = self.ViT_MLP(V_MLP, filter_num_MLP, activation, name='{}_mlp'.format(name))
      # Skip connection
      V_out = add([V_MLP, V_add], name='{}_skip_2'.format(name)) # <--- skip
      
      return V_out


  def transunet_2d_base(self, input_tensor, filter_num, stack_num_down=2, stack_num_up=2, 
                        embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                        activation='ReLU', mlp_activation='GELU', batch_norm=False, pool=True, unpool=True, 
                        backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='transunet'):
      '''
      The base of transUNET with an optional ImageNet-trained backbone.
      
      ----------
      Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L. and Zhou, Y., 2021. 
      Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306.
      
      Input
      ----------
          input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
          filter_num: a list that defines the number of filters for each \
                      down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                      The depth is expected as `len(filter_num)`.
          stack_num_down: number of convolutional layers per downsampling level/block. 
          stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
          activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
          batch_norm: True for batch normalization.
          pool: True or 'max' for MaxPooling2D.
                'ave' for AveragePooling2D.
                False for strided conv + batch norm + activation.
          unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                  'nearest' for Upsampling2D with nearest interpolation.
                  False for Conv2DTranspose + batch norm + activation.
          name: prefix of the created keras model and its layers.
          
          ---------- (keywords of ViT) ----------
          embed_dim: number of embedded dimensions.
          num_mlp: number of MLP nodes.
          num_heads: number of attention heads.
          num_transformer: number of stacked ViTs.
          mlp_activation: activation of MLP nodes.
          
          ---------- (keywords of backbone options) ----------
          backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                         None (default) means no backbone. 
                         Currently supported backbones are:
                         (1) VGG16, VGG19
                         (2) ResNet50, ResNet101, ResNet152
                         (3) ResNet50V2, ResNet101V2, ResNet152V2
                         (4) DenseNet121, DenseNet169, DenseNet201
                         (5) EfficientNetB[0-7]
          weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                   or the path to the weights file to be loaded.
          freeze_backbone: True for a frozen backbone.
          freeze_batch_norm: False for not freezing batch normalization layers.
          
      Output
      ----------
          X: output tensor.
      
      '''
      #activation_func = eval(activation)
      
      X_skip = []
      depth_ = len(filter_num)
      
      # ----- internal parameters ----- #
      
      # patch size (fixed to 1-by-1)
      patch_size = 1
      
      # input tensor size
      input_size = input_tensor.shape[1]
      
      # encoded feature map size
      encode_size = input_size // 2**(depth_-1)
      
      # number of size-1 patches
      num_patches = encode_size ** 2 
      
      # dimension of the attention key (= dimension of embedings)
      key_dim = embed_dim
      
      # number of MLP nodes
      filter_num_MLP = [num_mlp, embed_dim]
      
      # ----- UNet-like downsampling ----- #
      
      # no backbone cases
      if backbone is None:

          X = input_tensor

          # stacked conv2d before downsampling
          X = CONV_stack(X, filter_num[0], stack_num=stack_num_down, activation=activation, 
                         batch_norm=batch_norm, name='{}_down0'.format(name))
          X_skip.append(X)

          # downsampling blocks
          for i, f in enumerate(filter_num[1:]):
              X = UNET_left(X, f, stack_num=stack_num_down, activation=activation, pool=pool, 
                            batch_norm=batch_norm, name='{}_down{}'.format(name, i+1))        
              X_skip.append(X)

      # backbone cases
      else:
          # handling VGG16 and VGG19 separately
          if 'VGG' in backbone:
              backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
              # collecting backbone feature maps
              X_skip = backbone_([input_tensor,])
              depth_encode = len(X_skip)
              
          # for other backbones
          else:
              backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_-1, freeze_backbone, freeze_batch_norm)
              # collecting backbone feature maps
              X_skip = backbone_([input_tensor,])
              depth_encode = len(X_skip) + 1


          # extra conv2d blocks are applied
          # if downsampling levels of a backbone < user-specified downsampling levels
          if depth_encode < depth_:

              # begins at the deepest available tensor  
              X = X_skip[-1]

              # extra downsamplings
              for i in range(depth_-depth_encode):
                  i_real = i + depth_encode

                  X = UNET_left(X, filter_num[i_real], stack_num=stack_num_down, activation=activation, pool=pool, 
                                batch_norm=batch_norm, name='{}_down{}'.format(name, i_real+1))
                  X_skip.append(X)
          
      # subtrack the last tensor (will be replaced by the ViT output)
      X = X_skip[-1]
      X_skip = X_skip[:-1]

      # 1-by-1 linear transformation before entering ViT blocks
      X = Conv2D(filter_num[-1], 1, padding='valid', use_bias=False, name='{}_conv_trans_before'.format(name))(X)

      X = patch_extract((patch_size, patch_size))(X)
      X = patch_embedding(num_patches, embed_dim)(X)

      # stacked ViTs 
      for i in range(num_transformer):
          X = self.ViT_block(X, num_heads, key_dim, filter_num_MLP, activation=mlp_activation, 
                        name='{}_ViT_{}'.format(name, i))

      # reshape patches to feature maps
      X = tf.reshape(X, (-1, encode_size, encode_size, embed_dim))

      # 1-by-1 linear transformation to adjust the number of channels
      X = Conv2D(filter_num[-1], 1, padding='valid', use_bias=False, name='{}_conv_trans_after'.format(name))(X)

      X_skip.append(X)
      
      # ----- UNet-like upsampling ----- #
      
      # reverse indexing encoded feature maps
      X_skip = X_skip[::-1]
      # upsampling begins at the deepest available tensor
      X = X_skip[0]
      # other tensors are preserved for concatenation
      X_decode = X_skip[1:]
      depth_decode = len(X_decode)

      # reverse indexing filter numbers
      filter_num_decode = filter_num[:-1][::-1]

      # upsampling with concatenation
      for i in range(depth_decode):
          X = UNET_right(X, [X_decode[i],], filter_num_decode[i], stack_num=stack_num_up, activation=activation, 
                         unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i))

      # if tensors for concatenation is not enough
      # then use upsampling without concatenation 
      if depth_decode < depth_-1:
          for i in range(depth_-depth_decode-1):
              i_real = i + depth_decode
              X = UNET_right(X, None, filter_num_decode[i_real], stack_num=stack_num_up, activation=activation, 
                         unpool=unpool, batch_norm=batch_norm, concat=False, name='{}_up{}'.format(name, i_real))
              
      return X
 
  def create(self, num_classes, image_height, image_width, image_channels,
               base_filters = 16, num_layers = 6):
      print("==== TensorflowTransUNet.create ")
      input_size = (image_width, image_height, image_channels)

      self.filter_num     = self.config.get(MODEL, "filter_num", dvalue=[64, 128, 256, 512])
      self.stack_num_down = self.config.get(MODEL, "stack_num_down", dvalue=2)
      self.stack_num_up   = self.config.get(MODEL, "stack_num_up", dvalue=2)
      self.embed_dim      = self.config.get(MODEL, "embed_dim", dvalue=768)
      self.num_mlp        = self.config.get(MODEL, "num_mlp", dvalue=3072)
      self.num_heads      = self.config.get(MODEL, "num_heads", dvalue=12)
      self.num_transformer= self.config.get(MODEL, "num_transformer", dvalue=12)
      self.activation     = self.config.get(MODEL, "activation", dvalue='ReLU')
      self.mlp_activation = self.config.get(MODEL, "mlp_activation", dvalue='GELU')
      #self.output_activation='Softmax'
      self.batch_norm     = self.config.get(MODEL, "batch_norm", dvalue=False)
      self.pool           = self.config.get(MODEL, "pool", dvalue=True)
      self.unpool         = self.config.get(MODEL, "pool", dvalue=True) 
      self.backbone       = self.config.get(MODEL, "backbone", dvalue=None)
      self.weights        = self.config.get(MODEL, "weights", dvalue='imagenet')
      self.freeze_backbone= self.config.get(MODEL, "freeze_backbone", dvalue=True)
      self.freeze_batch_norm=self.config.get(MODEL, "freeze_batch_norm", dvalue=True)

      self.name='transunet'

      '''
      TransUNET with an optional ImageNet-trained bakcbone.
      
      
      ----------
      Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L. and Zhou, Y., 2021. 
      Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306.
      
      Input
      ----------
          input_size: the size/shape of network input, e.g., `(128, 128, 3)`.
          filter_num: a list that defines the number of filters for each \
                      down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                      The depth is expected as `len(filter_num)`.
          n_labels: number of output labels.
          stack_num_down: number of convolutional layers per downsampling level/block. 
          stack_num_up: number of convolutional layers (after concatenation) per upsampling level/block.
          activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
          output_activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                             Default option is 'Softmax'.
                             if None is received, then linear activation is applied.
          batch_norm: True for batch normalization.
          pool: True or 'max' for MaxPooling2D.
                'ave' for AveragePooling2D.
                False for strided conv + batch norm + activation.
          unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                  'nearest' for Upsampling2D with nearest interpolation.
                  False for Conv2DTranspose + batch norm + activation.                 
          name: prefix of the created keras model and its layers.
          
          ---------- (keywords of ViT) ----------
          embed_dim: number of embedded dimensions.
          num_mlp: number of MLP nodes.
          num_heads: number of attention heads.
          num_transformer: number of stacked ViTs.
          mlp_activation: activation of MLP nodes.
          
          ---------- (keywords of backbone options) ----------
          backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                         None (default) means no backbone. 
                         Currently supported backbones are:
                         (1) VGG16, VGG19
                         (2) ResNet50, ResNet101, ResNet152
                         (3) ResNet50V2, ResNet101V2, ResNet152V2
                         (4) DenseNet121, DenseNet169, DenseNet201
                         (5) EfficientNetB[0-7]
          weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                   or the path to the weights file to be loaded.
          freeze_backbone: True for a frozen backbone.
          freeze_batch_norm: False for not freezing batch normalization layers.
          
      Output
      ----------
          model: a keras model.
      
      '''
            
      IN = Input(input_size)
      
      # base    
      X = self.transunet_2d_base(IN, self.filter_num, stack_num_down=self.stack_num_down, 
                            stack_num_up=self.stack_num_up, 
                            embed_dim=self.embed_dim, num_mlp=self.num_mlp, num_heads=self.num_heads, 
                            num_transformer=self.num_transformer,
                            activation=self.activation, mlp_activation=self.mlp_activation, 
                            batch_norm=self.batch_norm, pool=self.pool, unpool=self.unpool,
                            backbone=self.backbone, weights=self.weights, freeze_backbone=self.freeze_backbone, 
                            freeze_batch_norm=self.freeze_batch_norm, name=self.name)
      
      # output layer
      output_activation = "Softmax"
      if num_classes == 1:
         output_activation = "Sigmoid"
      OUT = CONV_output(X, num_classes, kernel_size=1, activation=output_activation, name='{}_output'.format(self.name))
      
      # functional API model
      model = Model(inputs=[IN,], outputs=[OUT,], name='{}_model'.format(self.name))
      
      return model



if __name__ == "__main__":
  try:
    config_file = "./train_eval_inf.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
    model = TensorflowTransUNet(config_file)
    
  except:
    traceback.print_exc()
