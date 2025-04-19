import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch
from PIL import Image
from mpmath.libmp import normalize
from numpy.ma.core import min_val, max_val
from torchvision.transforms.v2.functional import to_pil_image

img_path = './image.jpg'
img_pil = Image.open(img_path)

img_np = img_pil.convert('RGB')
img_np = np.array(img_np)

transform = transforms.ToTensor()
img_tensor = transform(img_np).unsqueeze(0)

sobel_vertical = torch.tensor([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=torch.float32)

mean_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False, stride=1, groups=3)
mean_conv.weight.data = sobel_vertical.expand(3, 1, 3, 3)

with torch.no_grad():
    output_mean = mean_conv(img_tensor)

min_val = output_mean.min()
max_val = output_mean.max()

normalize_output = (output_mean - min_val) / (max_val - min_val)

to_pil_image = transforms.ToPILImage()
output_mean_np = to_pil_image(normalize_output.squeeze(0))
output_mean_np.show()
