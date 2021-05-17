import io
import numpy as np
from model import rcan
import utility


from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init

from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
# checkpoint = utility.checkpoint(args)

torch_model = rcan.RCAN(args)


model_url = 'https://www.dropbox.com/s/qm9vc0p0w9i4s0n/models_ECCV2018RCAN.zip?dl=0&file_subpath=%2Fmodels_ECCV2018RCAN%2FRCAN_BIX3.pt'
batch_size = 1    

map_location = lambda storage, loc: storage
 
torch_model.load

torch_model.eval()