import torch
import glob
import numpy as np
import os
import subprocess

import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from models.model import generate_model
from learner import Learner
from PIL import Image, ImageFilter

from PIL import Image, ImageFilter, ImageOps, ImageChops
import numpy as np
import torch
import random
import numbers
import pdb
import time
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
import argparse

try:
    import accimage
except ImportError:
    accimage = None


parser = argparse.ArgumentParser(description='Video Anomaly Detection')
parser.add_argument('--n', default='', type=str, help='file name')
args = parser.parse_args()


class ToTensor(object):

    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass
        

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass

#############################################################
#                        MAIN CODE                          #
#############################################################

model = generate_model() # feature extrctir
classifier = Learner().cuda() # classifier

checkpoint = torch.load('./weight/RGB_Kinetics_16f.pth')
model.load_state_dict(checkpoint['state_dict'])
checkpoint = torch.load('./weight/ckpt.pth')
classifier.load_state_dict(checkpoint['net'])

model.eval()
classifier.eval()

path = args.n + '/*'
save_path = args.n +'_result'
img = glob.glob(path)
img.sort()

segment = len(img)//16
x_value =[i for i in range(segment)]

inputs = torch.Tensor(1, 3, 16, 240, 320)
x_time = [jj for jj in range(len(img))]
y_pred = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

import os
import cv2

# Define the input and output directories
input_dir = '/path/to/input/directory'
output_dir = '/path/to/output/directory'

# Load the list of frames from the input directory
frames = os.listdir(input_dir)
frames.sort()

# Define the font and other text properties
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)  # Position of the text in the image
font_scale = 1
color = (0, 0, 255)  # Red color in BGR format
thickness = 2

# Loop over each frame and process it using the model
y_pred = []
for i, frame_name in enumerate(frames):
    # Load the current frame
    frame_path = os.path.join(input_dir, frame_name)
    frame = cv2.imread(frame_path)

    # Prepare the input tensor for the model
    inputs[:,:,:15,:,:] = inputs[:,:,1:,:,:]
    inputs[:,:,15,:,:] = ToTensor(1)(frame)
    inputs = inputs.cuda()

    # Run the input through the model and classifier to generate an output prediction
    start = time.time()
    output, feature = model(inputs)
    feature = F.normalize(feature, p=2, dim=1)
    out = classifier(feature)
    y_pred.append(out.item())
    end = time.time()

    # Calculate the FPS and convert the output prediction to a string
    FPS = str(1/(end-start))[:5]
    out_str = str(out.item())[:5]

    # Print the text onto the frame
    cv2.putText(frame, out_str, org, font, font_scale, color, thickness)

    # Save the resulting frame to the output directory
    output_name = f'output_{i}.jpg'
    output_path = os.path.join(output_dir, output_name)
    cv2.imwrite(output_path, frame)

