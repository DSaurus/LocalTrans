
from torch.utils.data.dataloader import DataLoader
import  lib.image.warping as warping
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import kornia
import os
import time

from lib.image.warping import *
from config.config import parse_config
from lib.model.localtrans import LocalTrans
from lib.data.dataset_homo import HomoDataset

import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':
    args = parse_config()
    
    os.makedirs('results/tb_log/%s' % args.name, exist_ok=True)
    os.makedirs('results/checkpoints/%s' % args.name, exist_ok=True)
    writer = SummaryWriter('results/tb_log/%s' % args.name, flush_secs=10)

    dataset = HomoDataset(args.dataroot, bias=args.random_bias,
                             downsample=args.downsample, random_color=args.random_color,
                             random_noise=args.random_noise, random_identity=False)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

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
    
    lr = args.lr
    optimizer1 = torch.optim.Adam(nets[0].parameters(), lr)
    optimizer2 = torch.optim.Adam(nets[1].parameters(), lr)
    optimizer3 = torch.optim.Adam(nets[2].parameters(), lr)
    optimizers = [optimizer1, optimizer2, optimizer3]

    epoch_idx = 0
    train_idx = 0
    if args.resume or args.resume_dir is not None:
        id = 0
        if args.resume_dir is not None:
            resume_dir = args.resume_dir
        else:
            resume_dir = os.path.join('results/checkpoints/%s' % args.name)
        for optim in optimizers:
            if os.path.isfile(os.path.join(resume_dir, 'optim%d_latest.pt' % id)):
                print('load from %s' % (os.path.join(resume_dir, 'optim%d_latest.pt' % id)))
                optim.load_state_dict(torch.load(os.path.join(resume_dir, 'optim%d_latest.pt' % id)))
            id += 1
        
        if os.path.isfile(os.path.join(resume_dir, 'state.pt')):
            state = torch.load(os.path.join(resume_dir, 'state.pt'))
            epoch_idx = state['epoch_idx']
            train_idx = state['train_idx']

    EPOCH = args.epoch

    training = True
    for epoch in range(epoch_idx, EPOCH):
        np.random.seed(int(time.time()))
        for data in dataloader:
            img1 = data["img1"].permute(0, 3, 1, 2).float().to(cuda)
            o_img2 = data["img2"].permute(0, 3, 1, 2).float().to(cuda)
            # imsave('show_dataset/img1.png', (img1.clamp(0,1)[:, :]*255).byte().cpu().permute(0, 2, 3, 1).numpy()[0, :, :, ::-1])
            # imsave('show_dataset/img2.png', (o_img2.clamp(0,1)[:, :, 32:32+128, 32:32+128]*255).byte().cpu().permute(0, 2, 3, 1).numpy()[0, :, :, ::-1])
            # exit(0)
            
            img2 = o_img2.clone()
            gt = data["gt"].reshape(-1, 2, 2, 2).to(cuda)

            result = []
            loss = 0
            for level in range(3):
                net = nets[level]
                flow = net(img1[:, :, 32:32+128, 32:32+128], img2[:, :, 32:32+128, 32:32+128], level)
                result.append(flow.clone())
                loss_level = F.l1_loss(flow.permute(0, 2, 3, 1), gt)
                writer.add_scalar('loss_%d' % level, loss_level.item())
                loss += loss_level
                if level != 2:
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
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            print(loss.item())
            if train_idx % 50 == 0:
                id = 0
                checkpoint_dir = os.path.join('results/checkpoints/%s' % args.name)
                for net in nets:
                    torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'net%d_latest.pt' % (id)))
                    torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'net%d_epoch_%d.pt' % (id, epoch)))
                    id += 1
                id = 0
                for optim in optimizers:
                    torch.save(optim.state_dict(), os.path.join(checkpoint_dir, 'optim%d_latest.pt' % (id)))
                    torch.save(optim.state_dict(), os.path.join(checkpoint_dir, 'optim%d_epoch_%d.pt' % (id, epoch)))
                    id += 1
                state = {"epoch_idx": epoch, "train_idx": train_idx}
                torch.save(state, os.path.join(checkpoint_dir, 'state.pt'))
                
                img2 = o_img2.clone()[:1]
                fig = plt.figure()
                for level in range(3):
                    net = nets[level]
                    W, H = 192, 192
                    B = 1
                    flow = result[level][:1]
                    
                    grid = gen_grid(2, 2, 32, 128, 32, 128, B).to(cuda)
                    grid_flow = grid + flow.permute(0, 2, 3, 1)
                    homo = kornia.geometry.find_homography_dlt(grid_flow.reshape(B, -1, 2).contiguous(), grid.reshape(B, -1, 2).contiguous())
                    img2 = kornia.geometry.warp_perspective(img2, homo, (192, 192))
                    plt.subplot(int('14%d' % (level+1)))
                    show_img1 = (img1.permute(0, 2, 3, 1)[0]).detach().cpu().numpy()[:, :, ::-1]
                    show_img2 = (img2.permute(0, 2, 3, 1)[0]).detach().cpu().numpy()[:, :, ::-1]
                    plt.imshow(show_img1 / 2 + show_img2 / 2)

                    # img2 = o_img2.clone()[:1]
                    # flow = gt.to(cuda)[:1].permute(0, 3, 1, 2)
                    # grid = gen_grid(2, 2, 32, 128, 32, 128, B).to(cuda)
                    # grid_flow = grid + flow.permute(0, 2, 3, 1)
                    # homo = kornia.geometry.find_homography_dlt(grid_flow.reshape(B, -1, 2).contiguous(), grid.reshape(B, -1, 2).contiguous())
                    # img2 = kornia.geometry.warp_perspective(img2, homo, (192, 192))
                    # plt.subplot(int('23%d' % (level+1+3)))
                    # show_img1 = (img1.permute(0, 2, 3, 1)[0]).detach().cpu().numpy()[:, :, ::-1]
                    # show_img2 = (img2.permute(0, 2, 3, 1)[0]).detach().cpu().numpy()[:, :, ::-1]
                    # plt.imshow(show_img1 / 2 + show_img2 / 2)
                plt.savefig('test.jpg')
                writer.add_figure("fig_epoch%d" % (epoch), fig, train_idx)
                plt.close('all')

            train_idx += 1
    
    # f = open('overfit_wo.txt', 'w')
    # for e in loss_list:
    #     f.write('%f\n' % e)