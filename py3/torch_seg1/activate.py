import numpy as np
import cv2
import torch
import torchvision
from models import SegNet1
import unets
import unets2

import time

#model = SegNet1()
#model = unets.UNet11()
#model = unets.UNet16()
#model = unets.AlbuNet()
#model = unets2.UNet11()
#model = unets2.UNet16()
model = unets2.LinkNet34()

#path2Img = '/home/daiver/R3DS/Kirill/FaceSegmentation1/eye_seg1/data/imgs/Img_2221495758_1.png'
path2Img = '/home/daiver/R3DS/Kirill/FaceSegmentation1/eye_seg1/data/imgs/Img_2164052578_1.png'
imgOrig = cv2.imread(path2Img)
img = imgOrig
img = torchvision.transforms.ToTensor()(img).view(1, img.shape[2], img.shape[0], img.shape[1])

print(img)


model.load_state_dict(torch.load('model.pt', 'cpu'))
model.eval()
#torch.onnx.export(model, img, "model.proto", verbose=True)

start = time.time()
output = torch.nn.functional.sigmoid(model(img))
end = time.time()
print('elapsed', end - start)
#print(output.shape, output.dtype)
#print(output)
#output = (output.detach().numpy() * 255).astype(np.uint8).reshape(output.shape[2], output.shape[3])


#overlay = imgOrig.copy().astype(np.float32)
#overlay[:, :, 1] += output.astype(np.float32)
#overlay[overlay > 255] = 255
#overlay = overlay.astype(np.uint8)

#cv2.imshow('output', output)
#cv2.imshow('orig', imgOrig)
#cv2.imshow('overlay', overlay)
#cv2.waitKey()
