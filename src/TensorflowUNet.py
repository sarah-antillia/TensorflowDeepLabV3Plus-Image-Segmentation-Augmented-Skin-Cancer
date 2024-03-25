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

# 2023/06/29 Updated create method to add BatchNormalization provied that 
#[model]
#normalization=True
# However, this True setting will not be recommended because this may have adverse effect
# on tiled_image_segmentation.

# 2023/07/01 Support Overlapped-Tiled-Image-Segmentation 
#[tiledinfer]
#overlapping=32
#Specify a pixel size to overlap-tiling.
#Specify 0 if you need no overlapping.

# 2023/11/01
# Remove set_seed method from TensorflowUNet class.

# 2023/11/01
# Added dropout_seed_fixing flag to [model] section
""" 
[model]
; 2023/11/01 Fixing a random-seed in Dropout layer
dropout_seed_fixing = True

if dropout_seed_fixing:
    u = Dropout(dropout_rate * f, seed=self.seed)(u)
"""

# 2023/11/01
# Added seedreset_callbacck flag to [train] section.
"""
; Experimental: Enable the random-seed-reset-callback if Ture.
; This will affect the behavior of Dropout layer of your CNN model.
seedreset_callback = True
"""

# 2023/11/01
# Added dataset_splitter flag to [train] section.
#; Enable splitting dataset into train and valid if True.
#dataset_splitter = True

"""
#; Enable splitting dataset into train and valid if True.
[train]
#dataset_splitter = True

# 2023/10/27

dataset_splitter = self.config.get(TRAIN, "dataset_splitter", dvalue=False) 

"""

import os
import sys
import datetime

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# 2023/10/20 "false" -> "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 2023/10/13: Added the following lines.
SEED = 137
os.environ['PYTHONHASHSEED']         = "0"

#os.environ['TF_DETERMINISTIC_OPS']   = '1'
#os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

print("=== os.environ['PYTHONHASHSEED']         {}".format(os.environ['PYTHONHASHSEED']))
# 2024/01/29
#print("=== os.environ['TF_DETERMINISTIC_OPS']   {}".format(os.environ['TF_DETERMINISTIC_OPS']))
#print("=== os.environ['TF_CUDNN_DETERMINISTIC'] {}".format(os.environ['TF_CUDNN_DETERMINISTIC']))

import shutil

import sys
import glob
import traceback
import random
import numpy as np
import cv2
from ConfigParser import ConfigParser

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from PIL import Image, ImageFilter, ImageOps

from tensorflow.keras.layers import Lambda

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D, Dropout, Conv2D, MaxPool2D, BatchNormalization

from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.activations import relu
from tensorflow.keras import Model
from tensorflow.keras.losses import  BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
#from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 2023/10/20
from tensorflow.python.framework import random_seed

from EpochChangeCallback import EpochChangeCallback
from GrayScaleImageWriter import GrayScaleImageWriter

#2023/10/26
from SeedResetCallback       import SeedResetCallback
from losses import dice_coef, basnet_hybrid_loss, sensitivity, specificity
# 2024/02/22 Added bce_dice_loss
from losses import iou_coef, iou_loss, bce_iou_loss, dice_loss,  bce_dice_loss

from mish import mish

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("=== GPU Name:", gpu.name, "  Type:", gpu.device_type)

# 2023/10/31
# See https://www.tensorflow.org/api_docs/python/tf/config/threading/set_intra_op_parallelism_threads
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# 2023/10/23
random.seed    = SEED
print("=== random.seed {}".format(SEED))

np.random.seed = SEED
print("=== numpy.random.seed {}".format(SEED))
tf.random.set_seed(SEED)
print("=== tf.random.set_seed({})".format(SEED))

# 2023/10/24
# See https://www.tensorflow.org/community/contribute/tests
# Always seed any source of stochasticity
random_seed.set_seed(SEED)
print("=== tensorflow.python.framework random_seed({})".format(SEED))

#2023/10/23
# Disable OpenCL and disable multi-threading.
#cv2.ocl.setUseOpenCL(False)
#cv2.setNumThreads(1)

cv2.setRNGSeed(SEED)
print("=== cv2.setRNGSeed ({})".format(SEED))

#See: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
#Module: tf.keras.metrics

#See also: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py

MODEL  = "model"
TRAIN  = "train"
INFER  = "infer"
EVAL   = "eval"
SEGMENTATION = "segmentation"

# 2023/06/10
TILEDINFER = "tiledinfer"

BEST_MODEL_FILE = "best_model.h5"


class TensorflowUNet:

  def __init__(self, config_file):
    #self.set_seed()
    self.seed        = SEED
    self.config_file = config_file
    self.config    = ConfigParser(config_file)
     
    self.config.dump_all()

    image_height   = self.config.get(MODEL, "image_height")
    image_width    = self.config.get(MODEL, "image_width")
    image_channels = self.config.get(MODEL, "image_channels")
    num_classes    = self.config.get(MODEL, "num_classes")
    base_filters   = self.config.get(MODEL, "base_filters")
    num_layers     = self.config.get(MODEL, "num_layers")
    # 2024/02/28: Added  
    activatation    = self.config.get(MODEL, "activation", dvalue="relu")
    self.activation = eval(activatation)
    print("=== activation {}".format(activatation))
    # 

    self.model     = self.create(num_classes, image_height, image_width, image_channels, 
                            base_filters = base_filters, num_layers = num_layers)
    
    learning_rate  = self.config.get(MODEL, "learning_rate")
    clipvalue      = self.config.get(MODEL, "clipvalue", 0.2)
    print("--- clipvalue {}".format(clipvalue))
  # 2023/11/10
    optimizer = self.config.get(MODEL, "optimizer", dvalue="Adam")
    if optimizer == "Adam":
      self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate,
         beta_1=0.9, 
         beta_2=0.999, 
         clipvalue=clipvalue,  #2023/06/26
         amsgrad=False)
      print("=== Optimizer Adam learning_rate {} clipvalue {} ".format(learning_rate, clipvalue))
    
    elif optimizer == "AdamW":
      # 2023/11/10  Adam -> AdamW (tensorflow 2.14.0~)
      self.optimizer = tf.keras.optimizers.AdamW(learning_rate = learning_rate,
         clipvalue=clipvalue,
         )
      print("=== Optimizer AdamW learning_rate {} clipvalue {} ".format(learning_rate, clipvalue))
            
    self.model_loaded = False

    # 2023/05/20 Modified to read loss and metrics from train_eval_infer.config file.
    binary_crossentropy = tf.keras.metrics.binary_crossentropy
    binary_accuracy     = tf.keras.metrics.binary_accuracy
    # Default loss and metrics functions
    self.loss    = binary_crossentropy
    self.metrics = [binary_accuracy]
    
    # Read a loss function name from our config file, and eval it.
    # loss = "binary_crossentropy"
    self.loss  = eval(self.config.get(MODEL, "loss"))

    # Read a list of metrics function names, and eval each of the list,
    # metrics = ["binary_accuracy"]
    metrics  = self.config.get(MODEL, "metrics")
    self.metrics = []
    for metric in metrics:
      self.metrics.append(eval(metric))
    
    print("--- loss    {}".format(self.loss))
    print("--- metrics {}".format(self.metrics))
    
    #self.model.trainable = self.trainable

    self.model.compile(optimizer = self.optimizer, loss= self.loss, metrics = self.metrics)
   
    show_summary = self.config.get(MODEL, "show_summary")
    if show_summary:
      self.model.summary()

    self.show_history = self.config.get(TRAIN, "show_history", dvalue=False)


  def create(self, num_classes, image_height, image_width, image_channels,
            base_filters = 16, num_layers = 5):
    # inputs
    print("Input image_height {} image_width {} image_channels {}".format(image_height, image_width, image_channels))
    inputs = tf.keras.layers.Input((image_height, image_width, image_channels))
    s= tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    
    # normalization is False on default.
    normalization = self.config.get(MODEL, "normalization", dvalue=False)
    print("--- normalization {}".format(normalization))
    # fixing_dropout_seed is False on default.
    dropout_seed_fixing = self.config.get(MODEL, "dropout_seed_fixing", dvalue=False)
    print("--- dropout_seed_fixing {}".format(dropout_seed_fixing))

    # Encoder
    dropout_rate = self.config.get(MODEL, "dropout_rate")
    enc         = []
    kernel_size = (3, 3)
    pool_size   = (2, 2)
    dilation    = (2, 2)
    strides     = (1, 1)
    # <experiment on="2023/06/20">
    # [model] 
    # Specify a tuple of base kernel size of odd number something like this: 
    # base_kernels = (5,5)
    base_kernels   = self.config.get(MODEL, "base_kernels", dvalue=(3,3))
    (k, k) = base_kernels
    kernel_sizes = []
    for n in range(num_layers):
      kernel_sizes += [(k, k)]
      k -= 2
      if k <3:
        k = 3
    rkernel_sizes =  kernel_sizes[::-1]
    rkernel_sizes = rkernel_sizes[1:] 
    
    # kernel_sizes will become a list [(7,7),(5,5), (3,3),(3,3)...] if base_kernels were (7,7)
    print("--- kernel_size   {}".format(kernel_sizes))
    print("--- rkernel_size  {}".format(rkernel_sizes))
    # </experiment>
    dilation = None
    try:
      dilation_ = self.config.get(MODEL, "dilation", (1, 1))
      (d1, d2) = dilation_
      if d1 == d2:
        dilation = dilation_
    except:
      pass

    dilations = []
    (d, d) = dilation
    for n in range(num_layers):
      dilations += [(d, d)]
      d -= 1
      if d <1:
        d = 1
    rdilations = dilations[::-1]
    rdilations = rdilations[1:]

    print("=== dilations  {}".format(dilations))
    print("=== rdilations {}".format(rdilations))

    for i in range(num_layers):
      filters = base_filters * (2**i)
      kernel_size = kernel_sizes[i] 
      dilation = dilations[i]
      
      print("--- kernel_size {}".format(kernel_size))
      print("--- dilation {}".format(dilation))
      
      c = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, 
                 activation=self.activation, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(s)
      # 2023/06/20
      if normalization:
        c = tf.keras.layers.BatchNormalization()(c) 

      # 2023/10/31
      if dropout_seed_fixing:
        c = tf.keras.layers.Dropout(dropout_rate * i, seed= self.seed)(c)
      else:
        c = tf.keras.layers.Dropout(dropout_rate * i)(c)

      c = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, 
                 activation=self.activation, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(c)
      # 2023/06/25
      if normalization:
        c = tf.keras.layers.BatchNormalization()(c) 
      if i < (num_layers-1):
        p = tf.keras.layers.MaxPool2D(pool_size=pool_size)(c)
        s = p
      enc.append(c)
    
    enc_len = len(enc)
    enc.reverse()
    n = 0
    c = enc[n]
    
    # --- Decoder
    for i in range(num_layers-1):
      kernel_size = rkernel_sizes[i] 
      dilation = rdilations[i]
      print("+++ kernel_size {}".format(kernel_size))
      print("+++ dilation {}".format(dilation))

      f = enc_len - 2 - i
      filters = base_filters* (2**f)
      u = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c)
      n += 1
      u = tf.keras.layers.concatenate([u, enc[n]])
      u = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, 
                 activation=self.activation, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(u)
      # 2023/06/20
      if normalization:
        u = tf.keras.layers.BatchNormalization()(u)
      # 2023/10/31 
      if dropout_seed_fixing:
        u = tf.keras.layers.Dropout(dropout_rate * f, seed=self.seed)(u)
      else:
        u = tf.keras.layers.Dropout(dropout_rate * f)(u)

      u = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, 
                 activation=self.activation, 
                 kernel_initializer='he_normal', dilation_rate=dilation, padding='same')(u)
      # 2023/06/25
      if normalization:
        u = tf.keras.layers.BatchNormalization()(u) 
      c  = u

    # outouts
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c)

    # create Model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    return model
  
  def create_dirs(self, eval_dir, model_dir ):
    # 2023/06/20
    dt_now = str(datetime.datetime.now())
    dt_now = dt_now.replace(":", "_").replace(" ", "_")
    create_backup = self.config.get(TRAIN, "create_backup", False)
    if os.path.exists(eval_dir):
      # if create_backup flag is True, move previous eval_dir to *_bak  
      if create_backup:
        moved_dir = eval_dir +"_" + dt_now + "_bak"
        shutil.move(eval_dir, moved_dir)
        print("--- Moved to {}".format(moved_dir))
      else:
        shutil.rmtree(eval_dir)

    if not os.path.exists(eval_dir):
      os.makedirs(eval_dir)

    if os.path.exists(model_dir):
      # if create_backup flag is True, move previous model_dir to *_bak  
      if create_backup:
        moved_dir = model_dir +"_" + dt_now + "_bak"
        shutil.move(model_dir, moved_dir)
        print("--- Moved to {}".format(moved_dir))      
      else:
        shutil.rmtree(model_dir)

    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

 
  #2023/08/20
  # Modified the second and the third parameter can be taken  
  # (train_generator, valid_generaator ) or (x_train_images,  y_train_smasks).
  def train(self, train_generator, valid_generator): 
    batch_size = self.config.get(TRAIN, "batch_size")
    epochs     = self.config.get(TRAIN, "epochs")
    patience   = self.config.get(TRAIN, "patience")
    eval_dir   = self.config.get(TRAIN, "eval_dir")
    model_dir  = self.config.get(TRAIN, "model_dir")
    #Modified to correct "save_weights_only" name
    save_weights_only = self.config.get(TRAIN, "save_weights_only", dvalue=False)

    metrics    = ["accuracy", "val_accuracy"]
    try:
      metrics    = self.config.get(TRAIN, "metrics")
    except:
      pass
    # 2023/06/20
    self.create_dirs(eval_dir, model_dir)
    # Copy current config_file to model_dir
    shutil.copy2(self.config_file, model_dir)
    print("-- Copied {} to {}".format(self.config_file, model_dir))
    
    weight_filepath   = os.path.join(model_dir, BEST_MODEL_FILE)

    reducer  = None
    lr_reducer = self.config.get(TRAIN, "learning_rate_reducer", dvalue=False )
    if lr_reducer:
      lr_patience = int(patience/2)
      if lr_patience == 0:
        lr_patience = 5
      lr_patience = lr_reducer = self.config.get(TRAIN, "reducer_patience", dvalue= lr_patience)
      reducer = ReduceLROnPlateau(
                        monitor = 'val_loss',
                        factor  = 0.1,
                        patience= lr_patience,
                        min_lr  = 0.0)

    early_stopping = EarlyStopping(patience=patience, verbose=1)
    check_point    = ModelCheckpoint(weight_filepath, verbose=1, 
                                     save_best_only=True,
                                     save_weights_only=save_weights_only)
    epoch_change   = EpochChangeCallback(eval_dir, metrics)
  
    if reducer:
      callbacks = [early_stopping, check_point, epoch_change, reducer]
    else:
      callbacks = [early_stopping, check_point, epoch_change]
   
    #2023/10/25
    seedreset_callback = self.config.get(TRAIN, "seedreset_callback", dvalue=False) 
    if seedreset_callback:
      print("=== Added SeedResetCallback")
      seedercb = SeedResetCallback(seed=self.seed)
      callbacks += [seedercb]
 
    if type(train_generator) == np.ndarray and type(valid_generator) == np.ndarray:
      x_train = train_generator
      y_train = valid_generator
 
      # 2023/10/27
      dataset_splitter = self.config.get(TRAIN, "dataset_splitter", dvalue=False) 
      print("=== Dataset_splitter {}".format(dataset_splitter))

      if dataset_splitter:
        """
        Split master dataset (x_train, y_train) into (train_x, train_y) and (valid_x, valid_y)
        This will help to improve the reproducibility of the model.
        """
        print("--- splitting the master dataset")
        train_size = int(0.8 * len(x_train)) 
        train_x = x_train[:train_size]
        train_y = y_train[:train_size]
        valid_x = x_train[train_size:]
        valid_y = y_train[train_size:]

        print("--- split the master into train(0.8) and valid(0.2)")
        print("=== Start model.fit ")
        history = self.model.fit(train_x, train_y, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_data= (valid_x, valid_y),
                    shuffle=False,
                    callbacks=callbacks,
                    verbose=1)
        self.visualiz_history(history)

      else:
        # By the parameter setting : validation_split=0.2,
        # x_train and y_train will be split into real_train (0.8) and 0.2 real_valid (0.2) 
        history = self.model.fit(x_train, y_train, 
                    validation_split=0.2, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    shuffle=False,
                    callbacks=callbacks,
                    verbose=1)
        self.visualiz_history(history)

    else:
      # train and valid dataset will be used by train_generator and valid_generator respectively
      steps_per_epoch  = self.config.get(TRAIN, "steps_per_epoch",  dvalue=400)
      validation_steps = self.config.get(TRAIN, "validation_steps", dvalue=800)
  
      history = self.model.fit(train_generator, 
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs, 
                    validation_data=valid_generator,
                    validation_steps= validation_steps,
                    shuffle = False,
                    callbacks=callbacks,
                    verbose=1)
      self.visualiz_history(history)

  def visualiz_history(self, history):
    history = history.history
    if self.show_history:
      keys = history.keys() 
      print("---history.keys {}".format(keys))
      #

  # 2023/05/09
  def load_model(self) :
    rc = False
    if  not self.model_loaded:    
      model_dir  = self.config.get(TRAIN, "model_dir")
      weight_filepath = os.path.join(model_dir, BEST_MODEL_FILE)
      if os.path.exists(weight_filepath):
        self.model.load_weights(weight_filepath)
        self.model_loaded = True
        print("=== Loaded a weight_file {}".format(weight_filepath))
        rc = True
      else:
        message = "Not found a weight_file " + weight_filepath
        raise Exception(message)
    else:
      pass
      #print("== Already loaded a weight file.")
    return rc
  
  # 2023/05/05 Added newly.    
  def infer(self, input_dir, output_dir, expand=True):
    colorize = self.config.get(SEGMENTATION, "colorize", dvalue=False)
    black    = self.config.get(SEGMENTATION, "black",    dvalue="black")
    white    = self.config.get(SEGMENTATION, "white",    dvalue="white")
    blursize = self.config.get(SEGMENTATION, "blursize", dvalue=None)

    writer       = GrayScaleImageWriter(colorize=colorize, black=black, white=white)

    # We are intereseted in png and jpg files.
    image_files  = glob.glob(input_dir + "/*.png")
    image_files += glob.glob(input_dir + "/*.jpg")
    image_files += glob.glob(input_dir + "/*.tif")
    #2023/05/15 Added *.bmp files
    image_files += glob.glob(input_dir + "/*.bmp")

    width        = self.config.get(MODEL, "image_width")
    height       = self.config.get(MODEL, "image_height")
    merged_dir   = None
    try:
      merged_dir = self.config.get(INFER, "merged_dir")
      if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
      if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    except:
      pass

    for image_file in image_files:
      basename = os.path.basename(image_file)
      name     = basename.split(".")[0]
      # <fixed> 2023/08/05 
      img      = cv2.imread(image_file)
      # </fixed>
      # img = BGR format 
      h = img.shape[0]
      w = img.shape[1]
      # Any way, we have to resize input image to match the input size of our TensorflowUNet model.
      img         = cv2.resize(img, (width, height))
      predictions = self.predict([img], expand=expand)
      prediction  = predictions[0]
      image       = prediction[0]    
      # Resize the predicted image to be the original image size (w, h), and save it as a grayscale image.
      # Probably, this is a natural way for all humans. 
      mask = writer.save_resized(image, (w, h), output_dir, name)

      print("--- image_file {}".format(image_file))
      if merged_dir !=None:
        # Resize img to the original size (w, h)
        img   = cv2.resize(img, (w, h))
        if blursize:
          img   = cv2.blur(img, blursize)
        img += mask
        merged_file = os.path.join(merged_dir, basename)
        cv2.imwrite(merged_file, img)

  def predict(self, images, expand=True):
    self.load_model()
    predictions = []
    for image in images:
      #print("=== Input image shape {}".format(image.shape))
      if expand:
        image = np.expand_dims(image, 0)
      pred = self.model.predict(image)
      predictions.append(pred)
    return predictions    

  def pil2cv(self, image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2: 
        pass
    elif new_image.shape[2] == 3: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

  def erode_image(self, img, shape_id=2, ksize=5):
    # shape_id takes a value in [0, 1, 2]
    #MORPH_RECT    = 0
    #MORPH_CROSS   = 1
    #MORPH_ELLIPSE = 2
    element = cv2.getStructuringElement(shape_id,
                       (ksize, ksize), 
                       ( -1, -1) )
      
    return cv2.erode(img, element)

  # 2023/06/05
  # 1 Split the original image to some tiled-images
  # 2 Infer segmentation regions on those images 
  # 3 Merge detected regions into one image
  # 2023/07/01
  # Added MARGIN to cropping 
  def infer_tiles(self, input_dir, output_dir, expand=True):
    
    image_files  = glob.glob(input_dir + "/*.png")
    image_files += glob.glob(input_dir + "/*.jpg")
    image_files += glob.glob(input_dir + "/*.tif")
    image_files += glob.glob(input_dir + "/*.bmp")
    MARGIN       = self.config.get(TILEDINFER, "overlapping", dvalue=0)
    print("MARGIN {}".format(MARGIN))

    merged_dir   = None
    try:
      merged_dir = self.config.get(TILEDINFER, "merged_dir")
      if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
      if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    except:
      pass

    width  = self.config.get(MODEL, "image_width")
    height = self.config.get(MODEL, "image_height")

    split_size  = self.config.get(TILEDINFER, "split_size", dvalue=width)
    print("---split_size {}".format(split_size))
    
    tiledinfer_debug = self.config.get(TILEDINFER, "debug", dvalue=False)
    tiledinfer_debug_dir = "./tiledinfer_debug_dir"
    if tiledinfer_debug:
      if os.path.exists(tiledinfer_debug_dir):
        shutil.rmtree(tiledinfer_debug_dir)
      if not os.path.exists(tiledinfer_debug_dir):
        os.makedirs(tiledinfer_debug_dir)
 
    # Please note that the default setting is "True".
    bitwise_blending  = self.config.get(TILEDINFER, "bitwise_blending", dvalue=True)
    bgcolor = self.config.get(TILEDINFER, "background", dvalue=0)  

    for image_file in image_files:
      image   = Image.open(image_file)
      w, h    = image.size

      # Resize the image to the input size (width, height) of our UNet model.      
      resized = image.resize((width, height))

      # Make a prediction to the whole image not tiled image of the image_file 
      cv_image= self.pil2cv(resized)
      predictions = self.predict([cv_image], expand=expand)
          
      prediction  = predictions[0]
      whole_mask  = prediction[0]    

      whole_mask  = self.normalize_mask(whole_mask)
      whole_mask  = cv2.resize(whole_mask, (w, h))
                
      basename = os.path.basename(image_file)
      self.tiledinfer_log = None
      
      if tiledinfer_debug and os.path.exists(tiledinfer_debug_dir):
        tiled_images_output_dir = os.path.join(tiledinfer_debug_dir, basename + "/images")
        tiled_masks_output_dir  = os.path.join(tiledinfer_debug_dir, basename + "/masks")
        if os.path.exists(tiled_images_output_dir):
          shutil.rmtree(tiled_images_output_dir)
        if not os.path.exists(tiled_images_output_dir):
          os.makedirs(tiled_images_output_dir)
        if os.path.exists(tiled_masks_output_dir):
          shutil.rmtree(tiled_masks_output_dir)
        if not os.path.exists(tiled_masks_output_dir):
          os.makedirs(tiled_masks_output_dir)
         
      w, h  = image.size

      vert_split_num  = h // split_size
      if h % split_size != 0:
        vert_split_num += 1

      horiz_split_num = w // split_size
      if w % split_size != 0:
        horiz_split_num += 1

      background = Image.new("L", (w, h), bgcolor)

      # Tiled image segmentation
      for j in range(vert_split_num):
        for i in range(horiz_split_num):
          left  = split_size * i
          upper = split_size * j
          right = left  + split_size
          lower = upper + split_size

          if left >=w or upper >=h:
            continue 
      
          left_margin  = MARGIN
          upper_margin = MARGIN
          if left-MARGIN <0:
            left_margin = 0
          if upper-MARGIN <0:
            upper_margin = 0

          right_margin = MARGIN
          lower_margin = MARGIN 
          if right + right_margin > w:
            right_margin = 0
          if lower + lower_margin > h:
            lower_margin = 0

          cropbox = (left  - left_margin,  upper - upper_margin, 
                     right + right_margin, lower + lower_margin )
          
          # Crop a region specified by the cropbox from the whole image to create a tiled image segmentation.      
          cropped = image.crop(cropbox)

          # Get the size of the cropped image.
          cw, ch  = cropped.size

          # Resize the cropped image to the model image size (width, height) for a prediction.
          cropped = cropped.resize((width, height))
          if tiledinfer_debug:
            #line = "image file {}x{} : x:{} y:{} width: {} height:{}\n".format(j, i, left, upper, cw, ch)
            #print(line)            
            cropped_image_filename = str(j) + "x" + str(i) + ".jpg"
            cropped.save(os.path.join(tiled_images_output_dir, cropped_image_filename))

          cvimage  = self.pil2cv(cropped)
          predictions = self.predict([cvimage], expand=expand)
          
          prediction  = predictions[0]
          mask        = prediction[0]    
          mask        = self.mask_to_image(mask)

          # Resize the mask to the same size of the corresponding the cropped_size (cw, ch)
          mask        = mask.resize((cw, ch))

          right_position = left_margin + width
          if right_position > cw:
             right_position = cw

          bottom_position = upper_margin + height
          if bottom_position > ch:
             bottom_position = ch

          # Excluding margins of left, upper, right and bottom from the mask. 
          mask         = mask.crop((left_margin, upper_margin, 
                                  right_position, bottom_position)) 
          iw, ih = mask.size
          if tiledinfer_debug:
            #line = "mask  file {}x{} : x:{} y:{} width: {} height:{}\n".format(j, i,  left, upper, iw, ih)
            #print(line)
            cropped_mask_filename = str(j) + "x" + str(i) + ".jpg"
            mask.save(os.path.join(tiled_masks_output_dir , cropped_mask_filename))
          # Paste the tiled mask to the background. 
          background.paste(mask, (left, upper))

      basename = os.path.basename(image_file)
      output_file = os.path.join(output_dir, basename)

      cv_background = self.pil2cv(background)
      bitwised = None
      if bitwise_blending:
        # Blend the non-tiled whole_mask and the tiled-backcround
        bitwised = cv2.bitwise_and(whole_mask, cv_background)
        bitwized_output_file =  os.path.join(output_dir, basename)
        cv2.imwrite(bitwized_output_file, bitwised)
      else:
        # Save the tiled-background. 
        background.save(output_file)

      print("=== Saved outputfile {}".format(output_file))

      if merged_dir !=None:
        img   = np.array(image)
        img   = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #2024/03/10
        if bitwise_blending:
          mask = bitwised
        else:
          mask  = cv_background 
 
        mask  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img += mask

        merged_file = os.path.join(merged_dir, basename)
        cv2.imwrite(merged_file, img)     

  def mask_to_image(self, data, factor=255.0):
    h = data.shape[0]
    w = data.shape[1]
    data = data*factor
    data = data.reshape([w, h])
    data = data.astype(np.uint8)
    image = Image.fromarray(data)
    image = image.convert("RGB")
    return image
  
  def normalize_mask(self, data, factor=255.0):
    h = data.shape[0]
    w = data.shape[1]
    data = data*factor
    data = data.reshape([w, h])
    data = data.astype(np.uint8)
    return data

  def evaluate(self, x_test, y_test): 
    self.load_model()
    # 2023/08/11
    batch_size = self.config.get(EVAL, "batch_size", dvalue=4)
    score = self.model.evaluate(x_test, y_test, 
                                batch_size = batch_size,
                                verbose = 1)
    print("Test loss    :{}".format(round(score[0], 4)))     
    print("Test accuracy:{}".format(round(score[1], 4)))
     

  # 2023/07/10
  def inspect(self, image_file='./model.png', summary_file="./summary.txt"):
    # Please download and install graphviz for your OS
    # https://www.graphviz.org/download/ 
    tf.keras.utils.plot_model(self.model, to_file=image_file, show_shapes=True)
    print("=== Saved model graph as an image_file {}".format(image_file))
    # https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
    with open(summary_file, 'w') as f:
      # Pass the file handle in as a lambda function to make it callable
      self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    print("=== Saved model summary as a text_file {}".format(summary_file))


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
    model    = TensorflowUNet(config_file)
    # Please download and install graphviz for your OS
    # https://www.graphviz.org/download/ 
    #image_file = './asset/model.png'
    #tf.keras.utils.plot_model(model.model, to_file=image_file, show_shapes=True)

  except:
    traceback.print_exc()
    
