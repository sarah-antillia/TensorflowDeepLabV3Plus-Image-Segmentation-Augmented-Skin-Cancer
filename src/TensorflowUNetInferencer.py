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

# TensorflowUNetInfer.py
# 2023/06/05 to-arai


import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import shutil
import sys
import traceback

from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset

from TensorflowUNet import TensorflowUNet

from TensorflowAttentionUNet import TensorflowAttentionUNet 
from TensorflowEfficientUNet import TensorflowEfficientUNet
from TensorflowMultiResUNet import TensorflowMultiResUNet
from TensorflowSwinUNet import TensorflowSwinUNet
# 2023/12/10 Added the follwoing line
from TensorflowTransUNet import TensorflowTransUNet

from TensorflowUNet3Plus import TensorflowUNet3Plus
from TensorflowU2Net import TensorflowU2Net
#from TensorflowDeepLabV3Plus import TensorflowDeepLabV3Plus
from TensorflowSharpUNet import TensorflowSharpUNet
#from TensorflowBASNet    import TensorflowBASNet
from TensorflowDeepLabV3Plus import TensorflowDeepLabV3Plus

MODEL  = "model"
TRAIN  = "train"
INFER  = "infer"

if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    config     = ConfigParser(config_file)
    images_dir = config.get(INFER, "images_dir")
    output_dir = config.get(INFER, "output_dir")
 
    # Create a UNetMolde and compile
    #model          = TensorflowUNet(config_file)
    ModelClass = eval(config.get(MODEL, "model", dvalue="TensorflowUNet"))
    print("=== ModelClass {}".format(ModelClass))

    model     = ModelClass(config_file)

    if not os.path.exists(images_dir):
      raise Exception("Not found " + images_dir)
    
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    model.infer(images_dir, output_dir, expand=True)

  except:
    traceback.print_exc()
    

