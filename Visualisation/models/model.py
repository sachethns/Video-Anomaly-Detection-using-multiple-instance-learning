from __future__ import division
import torch
from torch import nn
from models import resnext
import pdb

def generate_model():
    model = resnext.resnet101(
            num_classes=400,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=16,
            input_channels=3,
            output_layers=[])
    

    model = model.cuda()
    model = nn.DataParallel(model)

    return model

