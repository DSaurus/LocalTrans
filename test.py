from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import  lib.image.warping as warping
from lib.model.localtrans import LocalTrans
from lib.data.dataset_homo import *
from lib.image.warping import *
from config.config import parse_config


class PSNR(nn.Module):
    def __init__(self, max_val=1., mode='Y'):
        super(PSNR, self).__init__()
        self.max_val = max_val
        self.mode = mode

    def forward(self, x, y):
        if self.mode == 'Y' and x.shape[1] == 3 and y.shape[1] == 3:
            x = kornia.color.bgr_to_grayscale(x)
            y = kornia.color.bgr_to_grayscale(y)
        mse = F.mse_loss(x, y, reduction='mean')
        psnr = 10 * torch.log10(self.max_val ** 2 / mse)
        return psnr

class SSIM(nn.Module):
    def __init__(self, window_size=11):
        super(SSIM, self).__init__()
        self.window_size = window_size

    def forward(self, x, y):
        if x.shape[1] == 3:
            x = kornia.color.bgr_to_grayscale(x)
        if y.shape[1] == 3:
            y = kornia.color.bgr_to_grayscale(y)
        return 1 - kornia.losses.ssim(x, y, self.window_size, 'mean')

def mkdir(dir):
    try:
        os.mkdir(dir)
    except Exception:
        pass


if __name__ == '__main__':
    args = parse_config()
    dataset = HomoDataset('/media/data1/shaoruizhi/AttentionWarping/val2014', bias=args.random_bias,
                             downsample=args.downsample, random_color=args.random_color,
                             random_noise=args.random_noise, random_identity=False)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)

    net1 = LocalTrans()
    net2 = LocalTrans()
    net3 = LocalTrans()
    nets = [net1, net2, net3]

    cuda = torch.device('cuda:%d' % args.gpu_id)
    torch.cuda.set_device(args.gpu_id)

    if args.resume or args.resume_dir is not None:
        id = 0
        if args.resume_dir is not None:
            resume_dir = args.resume_dir
        else:
            resume_dir = os.path.join('results/checkpoints/%s' % args.name)
        for net in nets:
            if os.path.isfile(os.path.join(resume_dir, 'net%d_latest.pt' % id)):
                print('load from %s' % (os.path.join(resume_dir, 'net%d_latest.pt' % id)))
                net.load_state_dict(torch.load(os.path.join(resume_dir, 'net%d_latest.pt' % id)))
            id += 1

    for net in nets:
        net.to(cuda)
        net.eval()

    np.random.seed(int(time.time()))
    train_idx = 0
    loss = 0
    
    loss_list = {}
    loss_list[0] = []
    loss_list[1] = []
    loss_list[2] = []
    loss_list[3] = []
    total_psnr = 0
    total_ssim = 0
    idx = 0
    import time
    total_time = 0
    for data in tqdm(dataloader):
        idx += 1
        img1 = data["img1"].permute(0, 3, 1, 2).float().to(cuda)
        o_img1 = img1.clone()
        o_img2 = data["img2"].permute(0, 3, 1, 2).float().to(cuda)
        hr_img = data["hr_img"].float().to(cuda)
        img2 = o_img2.clone()
        gt = data["gt"].reshape(-1, 2, 2, 2).to(cuda)
        o_gt = data["gt"].reshape(-1, 2, 2, 2).to(cuda).clone()

        result = []
        t = time.time()
        with torch.no_grad():
            for level in range(0, 3):
                net = nets[level]
                flow = net(img1[:, :, 32:32+128, 32:32+128], img2[:, :, 32:32+128, 32:32+128], level)
                result.append(flow.clone())
                loss_level = torch.mean(torch.sqrt(torch.sum((flow.permute(0, 2, 3, 1) - gt)**2, dim=3)))
                
                if level == 2:
                    loss += loss_level
                dis = torch.sqrt(torch.sum((flow.permute(0, 2, 3, 1) - gt)**2, dim=3)).reshape(-1, 4)
                dis = torch.mean(dis, dim=1)
                loss_list[level].append(dis)
                # if level != 2:
                B = img1.shape[0]
                grid = gen_grid(2, 2, 32, 128, 32, 128, B).to(cuda)
                grid_flow = grid + flow.permute(0, 2, 3, 1)
                homo = kornia.geometry.find_homography_dlt(grid_flow.reshape(B, -1, 2).contiguous(), grid.reshape(B, -1, 2).contiguous())
                img2 = kornia.geometry.warp_perspective(img2, homo, (192, 192)).detach()

                ori_grid = gen_grid(2, 2, 32, 128, 32, 128, B, device=cuda)
                grid = ori_grid + gt
                grid3 = torch.cat([grid.reshape(B, -1, 2), torch.ones(B, 4, 1, device=cuda)], dim=2)
                grid3 = (homo @ grid3.transpose(1, 2)).transpose(1, 2)
                grid2 = grid3[:, :, :2] / grid3[:, :, 2:]
                gt = (grid2.reshape(B, 2, 2, 2) - ori_grid).detach()
                print('level%d:' % level, loss_level)
                # if level == 2:
                #     break
        total_time += time.time() - t
        train_idx += 1
        
    for l in range(3):
        loss_list[l] = torch.cat(loss_list[l], dim=0).detach().cpu().numpy()
    
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.grid(True, 'both')
    for l in range(3):
        a = 1e-2
        x = []
        y = []
        while True:
            for i in range(1, 10):
                x.append(a*i)
                y.append(np.sum(loss_list[l] < a*i) / len(loss_list[l]))
            a *= 10
            if a >= 100:
                break
        print(y)
        ax.plot(x, y)
    plt.savefig('data.jpg')
    print(loss)