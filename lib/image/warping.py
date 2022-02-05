import torch.nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch

def homo_grid(homo, grid):
    device = grid.device
    B = grid.shape[0]
    grid = grid.reshape(B, -1, 2)
    grid3 = torch.cat([grid, torch.ones(B, grid.shape[1], 1, device=device)], dim=2)
    grid3 = (homo @ grid3.transpose(1, 2)).transpose(1, 2)
    grid2 = grid3[:, :, :2] / grid3[:, :, 2:]
    return grid2

def resize_homo(homo, src, tgt):
    # W, H
    scale1 = np.eye(3)
    scale1[0, 0] = src[0] / tgt[0]
    scale1[1, 1] = src[1] / tgt[1]
    scale2 = np.eye(3)
    scale2[0, 0] = tgt[0] / src[0]
    scale2[1, 1] = tgt[1] / src[1]
    return scale2 @ homo @ scale1

def downsample_kitti_flow(flow):
    H, W, _ = flow.shape
    flow_new = np.zeros((H // 2, W // 2, 2), dtype=float)
    flow_count = np.zeros((H // 2, W // 2, 1), dtype=int)
    for r in range(H):
        for c in range(W):
            flow_new[r // 2, c // 2, :] += flow[r, c, :]
            if flow[r, c, 0] != 0 and flow[r, c, 1] != 0:
                flow_count[r // 2, c // 2, 0] += 1
    for r in range(H // 2):
        for c in range(W // 2):
            if flow_count[r, c, 0] > 0:
                flow_new[r, c, :] /= flow_count[r, c, 0]
    return flow_new

def sample_upsample(sample, factor=2, size=None):
    """
    :param sample: [B, H, W, 2]
    :param factor: up_sample factor
    :param size: up_sample size
    :return: upsampled sample
    """
    sample = sample.permute(0, 3, 1, 2)
    if size is None:
        sample = F.upsample_bilinear(sample, scale_factor=factor)
    else:
        sample = F.upsample_bilinear(sample, size=size)
    return sample.permute(0, 2, 3, 1)

def sample_to_grid(sample, width, height):
    """
    normalized to [Width, height]
    :param grid: [B, H, W, 2]
    :param width: grid width
    :param height: grid height
    :return: sample: [B, H, W, 2]
    """
    sample = (sample + 1) / 2
    sample[:, :, :, 0] *= width
    sample[:, :, :, 1] *= height
    return sample

def grid_to_sample(grid, width, height):
    """
    normalized to [-1, 1]
    :param grid: [B, H, W, 2]
    :param width: grid width
    :param height: grid height
    :return: sample: [B, H, W, 2]
    """
    grid[:, :, :, 0] /= width
    grid[:, :, :, 1] /= height
    grid = (grid - 0.5) * 2
    return grid

def gen_grid(cols, rows, w_s, width, h_s, height, batch, device=None):
    """
    :param cols: grid cols
    :param rows: grid rows
    :param w_s:  grid x start
    :param width: gird width
    :param h_s:  grid y start
    :param height: grid height
    :param batch: batch size
    :return: grid [B, H, W, 2]
    """
    # return grid[B, H, W, 2]
    if device is None:
        y, x = torch.meshgrid([torch.linspace(h_s, h_s + height, rows), torch.linspace(w_s, w_s + width, cols)])
    else:
        y, x = torch.meshgrid([torch.linspace(h_s, h_s + height, rows, device=device), torch.linspace(w_s, w_s + width, cols, device=device)])
    grid = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], 1)
    grid = grid.reshape((rows, cols, 2))
    grid = grid.unsqueeze(0)
    grid = grid.repeat((batch, 1, 1, 1))
    return grid

def gen_corr_mask(W, H, radius, new_W, new_H, B):
    x, y = torch.meshgrid([torch.linspace(0.5, W - 0.5, new_W), torch.linspace(0.5, H - 0.5, new_H)])
    grid = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), torch.ones(new_W * new_H, 1)], dim=1)
    ori_grid = (grid[:, :2]).reshape((new_H, new_W, 2))

    space = W // new_W
    ind = 0

    k_r = radius // 2
    mask_grid = np.ones((radius * radius, new_H, new_W))
    for y in np.linspace(-k_r, k_r, radius):
        for x in np.linspace(-k_r, k_r, radius):
            t_grid = ori_grid.clone()
            t_grid[:, :, 0] += x * space
            t_grid[:, :, 1] += y * space
            mask_grid[ind][t_grid[:, :, 0] < 0] = 0
            mask_grid[ind][t_grid[:, :, 0] > W] = 0
            mask_grid[ind][t_grid[:, :, 1] < 0] = 0
            mask_grid[ind][t_grid[:, :, 1] > H] = 0
            ind += 1
    mask_grid = torch.from_numpy(mask_grid).float().unsqueeze(0).repeat(B, 1, 1, 1)
    return mask_grid

def gen_corr_grid(W, H, radius, new_W, new_H, shift_grid, mask_grid):
    B = shift_grid.shape[0]
    space = W // new_W
    ind = 0
    k_r = radius // 2
    cor_grid = torch.zeros((B, radius * radius, new_H, new_W)).to(shift_grid.device)

    for y in np.linspace(-k_r, k_r, radius):
        for x in np.linspace(-k_r, k_r, radius):
            cor_grid[:, ind, :, :] = - (
                        (shift_grid[:, 0, :, :] / space - x) ** 2 + (shift_grid[:, 1, :, :] / space - y) ** 2) / radius
            ind += 1
    cor_grid = cor_grid * mask_grid - (1 - mask_grid) * 1e6
    cor_grid = torch.softmax(cor_grid, dim=1)

    return cor_grid

def gen_weight_map(width, height, cols, rows):
    Y, X = torch.linspace(0, 1, height), torch.linspace(0, 1, width)
    block_width = 1 / (cols-1)
    block_height = 1 / (rows-1)
    weight = torch.zeros((width*height, rows*cols))
    ind = 0
    for y in Y:
        for x in X:
            r = max(0, min(rows-2, (int)(y / block_height)))
            c = max(0, min(cols-2, (int)(x / block_width)))
            tx = (x - c*block_width) / block_width
            ty = (y - r*block_height) / block_height
            weight[ind, r*cols + c] = (1-tx)*(1-ty)
            weight[ind, r * cols + c+1] = tx * (1 - ty)
            weight[ind, (r+1) * cols + c] = (1 - tx) * ty
            weight[ind, (r+1) * cols + (c+1)] = tx*ty
            ind += 1
    return weight


def warping_batch_tensor_Npts(img_tensor, pts, weight, rows, cols):
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    weight = weight.to(device)

    y, x = torch.meshgrid([torch.linspace(-1, 1, rows), torch.linspace(-1, 1, cols)])
    y, x = y.flatten(), x.flatten()
    ptsN = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], dim=1).to(device)
    ptsN = pts.reshape((B, rows*cols, 2)) / 10 + ptsN.unsqueeze(0)
    grid = weight.unsqueeze(0) @ ptsN
    grid = grid.reshape((-1, H, W, 2))
    img = F.grid_sample(img_tensor, grid, mode='bilinear')
    return img

def warping_batch_tensor_4pts(img_tensor, pts):
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    y, x = torch.meshgrid([torch.linspace(0, 1, H), torch.linspace(0, 1, W)])
    y, x = y.flatten(), x.flatten()
    weight = torch.from_numpy(np.zeros((H*W, 4), dtype=float)).float()
    weight[:, 0] = (1-x)*(1-y)
    weight[:, 1] = x*(1-y)
    weight[:, 2] = (1-x)*y
    weight[:, 3] = x*y
    weight = weight.to(device)
    pts4 = torch.from_numpy(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=float)).float().to(device)
    pts4 = pts.reshape((B, 4, 2)) / 10 + pts4.unsqueeze(0)
    grid = weight.unsqueeze(0) @ pts4
    grid = grid.reshape((-1, H, W, 2))
    img = F.grid_sample(img_tensor, grid, mode='bilinear')
    return img


def warping_batch_tensor(img_tensor, transform):
    # img_tensor [B, C, H, W]
    # transform [B, 3, 3]
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    x, y = torch.meshgrid([torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)])
    x, y = x.flatten(), y.flatten()
    grid = torch.cat([x.reshape((-1, 1)), y.reshape((-1, 1))], dim=1)
    grid = grid.reshape((H, W, 2))
    grid = torch.cat([grid, torch.ones((H, W, 1))], 2)

    grid = grid.reshape((-1, 3)).transpose(1, 0).unsqueeze(0).to(device)
    grid = transform.matmul(grid)
    grid = grid[:, :2, :] / (grid[:, 2, :].unsqueeze(1) + 1e-9)
    grid = grid.permute(0, 2, 1).reshape(-1, H, W, 2)
    return F.grid_sample(img_tensor, grid, mode='bilinear').transpose(2, 3)

def crop_image(img_tensor, ratio):
    # img_tensor B, C, W, H
    B, C, W, H = img_tensor.shape
    out = img_tensor[:, :, (int)(H*(1-ratio)/2) : (int)(H*(1+ratio)/2), (int)(W*(1-ratio)/2) : (int)(W*(1+ratio)/2)]
    return out