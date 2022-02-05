import torch.utils.data.dataset
import cv2
import os
import numpy as np
import kornia
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import ColorJitter, ToTensor, ToPILImage
from PIL import Image, ImageOps
from lib.image.warping import *
from lib.image.flow import *
import time
import math

class HomoDataset(torch.utils.data.Dataset):
    def __init__(self, _image_dir, testing=False, bias=0.3, size=256,
                downsample=8, random_color=True, random_noise=True, random_identity=True, show=False):
        super(HomoDataset, self).__init__()

        self.IMAGE_DIR = _image_dir
        self.images_name = os.listdir(_image_dir)
        if show:
            self.images_name.sort()
            self.images_name[0] = self.images_name[33]
        self.show = show
        

        self.ave_img = np.array([103.82500679, 114.0616572,  119.89195349])
        self.size = (size, size)
        self.bias = bias
        self.downsample = downsample
        self.random_color = random_color
        self.random_noise = random_noise
        self.random_identity = random_identity

    def __getitem__(self, index):
        if self.show and index == 0:
            torch.random.manual_seed(3354)
            np.random.seed(3354)
            print(np.random.rand())
        sample_image_name = self.images_name[index]
        sample_image = cv2.imread(os.path.join(self.IMAGE_DIR, sample_image_name))
        to_tensor = ToTensor()
        to_PIL = ToPILImage()
        sample_image = to_tensor(sample_image)
        C, H, W = sample_image.shape
        if H > W:
            sample_image = F.interpolate(sample_image.unsqueeze(0), size=(240, int(240*H/W)), mode='bilinear').squeeze(0)
        else:
            sample_image = F.interpolate(sample_image.unsqueeze(0), size=(int(240*W/H), 240), mode='bilinear').squeeze(0)
        C, H, W = sample_image.shape
        y, x = int(np.random.rand()*(H-192)), int(np.random.rand()*(W-192))
        sample_image = sample_image[:, y:y+192, x:x+192]

        corner1 = np.array([(32, 32), (128+32, 32), (32, 128+32), (128+32, 128+32)])
        corner2 = np.zeros(corner1.shape)
        diff = []
        while True:
            re_random = False
            for j in range(4):
                dx = np.random.rand() * 64 - 32
                dy = np.random.rand() * 64 - 32
                # dx = np.random.rand() * 16 - 8
                # dy = np.random.rand() * 16 - 8
                corner2[j] = corner1[j] + (dx, dy)
            corner3 = np.zeros(corner1.shape)
            corner3[0] = corner2[0]
            corner3[1] = corner2[1]
            corner3[2] = corner2[3]
            corner3[3] = corner2[2]
            for j in range(4):
                a, b, c = corner3[j], corner3[(j+1)%4], corner3[(j+2)%4]
                ba = a - b
                bc = c - b
                cos = np.sum(ba*bc) / np.sqrt(np.sum(ba*ba) * np.sum(bc*bc))
                theta = math.acos(cos)
                if theta > 3/4*math.acos(-1):
                    re_random = True
            if not re_random:
                break
        
        h, _ = cv2.findHomography(corner1, corner2)
        h = torch.from_numpy(h).unsqueeze(0).float()
        sample_image *= 255
        test_img = sample_image.unsqueeze(0).float()
        test_img = kornia.geometry.warp_perspective(test_img, h, (192, 192))
        ori_test_img = test_img.clone().squeeze(0) / 255
        test_img = F.interpolate(F.interpolate(test_img, size=(192 // self.downsample, 192 // self.downsample), mode='bilinear'), size=(192, 192), mode='bilinear').squeeze(0)
        test_img[test_img > 255] = 255
        test_img[test_img < 0] = 0
        sample_image[sample_image > 255] = 255
        sample_image[sample_image < 0] = 0
        test_img = Image.fromarray(test_img.permute(1, 2, 0).byte().numpy())
        sample_image = Image.fromarray(sample_image.permute(1, 2, 0).byte().numpy())

        if self.random_color:
            select = np.random.rand()
            cj = ColorJitter(0.5, 0.5, 0.5)
            if self.show and index == 0:
                select = 0
            if select > 0.5:
                sample_image = cj(sample_image)
            else:
                test_img = cj(test_img)

        gt = torch.FloatTensor(corner2 - corner1)
        sample_image = to_tensor(sample_image).permute(1, 2, 0)
        test_img = to_tensor(test_img).permute(1, 2, 0)
        
        if self.random_noise:
            sample_image = sample_image + torch.FloatTensor(np.random.normal(scale=0.02, size=(192, 192, 3)))
            test_img = test_img + torch.FloatTensor(np.random.normal(scale=0.02, size=(192, 192, 3)))

        res = {
            "img1": sample_image,
            "img2": test_img,
            "hr_img": ori_test_img,
            "gt": gt
        }
        return res

    def __len__(self):
        return self.images_name.__len__()