import torch
import lib.image.warping as warping
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2

def torch2numpy(img):
    img = img.detach().cpu()
    img = img.squeeze(0).permute(1, 2, 0).numpy()
    return img

def estimate_flow_homography(flow, W, H):
    rows = flow.shape[1]
    cols = flow.shape[2]
    grid = warping.gen_grid(cols, rows, 0.5, W-0.5, 0.5, H-0.5, 1)
    flow_grid = grid + flow
    homo, _ = cv2.findHomography(grid.reshape((-1, 2)).numpy(), flow_grid.reshape((-1, 2)).numpy(), 0)
    return homo

def estimate_flow(net, img1, img2, start_i=0):
    H, W, C = img1.shape
    sample_data = warping.gen_grid(2 * (2 ** start_i), 2 * (2 ** start_i), 0.5, H - 0.5, 0.5, W - 0.5, 1)
    sample_data = warping.grid_to_sample(sample_data, W, H).cuda()
    net_img1 = img1.unsqueeze(0).permute(0, 3, 1, 2).cuda()
    net_img2 = img2.unsqueeze(0).permute(0, 3, 1, 2).cuda()
    flow = 0
    for i in range(start_i, 4):
        new_flow, sample_data_new, _ = net.forward(net_img1, net_img2, sample_data, i)
        flow = flow + new_flow
        if i == 3:
            break
        flow = warping.sample_upsample(flow)
        sample_data = sample_data_new

    return flow.detach().cpu().numpy()

def estimate_warp(net, img1, img2, start_i=0):
    H, W, C = img1.shape
    sample_data = warping.gen_grid(2 * (2**start_i),  2 * (2**start_i), 0.5, H - 0.5, 0.5, W - 0.5, 1)
    sample_data = warping.grid_to_sample(sample_data, W, H).cuda()
    net_img1 = (torch.FloatTensor(img1) - 128) / 255
    net_img2 = (torch.FloatTensor(img2) - 128) / 255
    net_img1 = net_img1.unsqueeze(0).permute(0, 3, 1, 2).cuda()
    net_img2 = net_img2.unsqueeze(0).permute(0, 3, 1, 2).cuda()
    flow = 0
    for i in range(start_i, 4):
        new_flow, sample_data_new, _ = net.forward(net_img1, net_img2, sample_data, i)
        flow = flow + new_flow
        flow = warping.sample_upsample(flow)
        sample_data = sample_data_new
    grid = warping.gen_grid(W, H, 0.5, W-0.5, 0.5, H-0.5, 1).cuda()
    flow = warping.sample_upsample(flow, size=(H, W))

    sample_data = warping.sample_upsample(sample_data, size=(H, W))

    sample_flow = grid + flow
    sample_flow = warping.grid_to_sample(sample_flow, W, H)
    sample_img2 = F.grid_sample(net_img2, sample_data)

    # show_img1 = torch2numpy(net_img1 / 2 + net_img2 / 2)
    # show_img2 = torch2numpy(net_img1 / 2 + sample_img2 / 2)
    # plt.subplot(223)
    # plt.imshow(torch2numpy(net_img1 + 0.5)[:, :, ::-1])
    # plt.subplot(224)
    # plt.imshow(torch2numpy(net_img2 + 0.5)[:, :, ::-1])
    # plt.subplot(221)
    # plt.imshow( (show_img1+0.5)[:, :, ::-1])
    # plt.subplot(222)
    # plt.imshow((show_img2 + 0.5)[:, :, ::-1])
    # plt.show()
    return flow.detach().cpu().numpy()