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

# TensorflowUNetGeneratorTrainer.py
# 2023/08/20 to-arai
# 2023/12/10 Updated

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import shutil
import sys
import traceback

from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset

from TensorflowUNet import TensorflowUNet

from ImageMaskDatasetGenerator import ImageMaskDatasetGenerator

from TensorflowAttentionUNet import TensorflowAttentionUNet 
from TensorflowEfficientUNet import TensorflowEfficientUNet
from TensorflowMultiResUNet import TensorflowMultiResUNet
from TensorflowSwinUNet import TensorflowSwinUNet

# 2023/12/10 Added the follwoing line
from TensorflowTransUNet import TensorflowTransUNet

from TensorflowUNet3Plus import TensorflowUNet3Plus
from TensorflowU2Net import TensorflowU2Net
# 2024/03/20 Added the following line
from TensorflowSharpUNet import TensorflowSharpUNet
#from TensorflowBASNet    import TensorflowBASNet
from TensorflowDeepLabV3Plus import TensorflowDeepLabV3Plus

MODEL  = "model"
TRAIN  = "train"
EVAL   = "eval"

if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    config   = ConfigParser(config_file)

    # Create a UNetModel and compile
    ModelClass = eval(config.get(MODEL, "model", dvalue="TensorflowUNet"))
    print("=== ModelClass {}".format(ModelClass))
    model     = ModelClass(config_file)
        
    train_gen = ImageMaskDatasetGenerator(config_file, dataset=TRAIN)
    train_generator = train_gen.generate()

    valid_gen = ImageMaskDatasetGenerator(config_file, dataset=EVAL)
    valid_generator = valid_gen.generate()

    model.train(train_generator, valid_generator)

  except:
    traceback.print_exc()
    
