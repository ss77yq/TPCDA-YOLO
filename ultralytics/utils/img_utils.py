import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as SSIM


def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps


def torchSSIM(img1, img2):
    # Convert torch tensors to numpy arrays
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()

    # Calculate SSIM
    ssim = SSIM(img1, img2, channel_axis=2)
    return ssim


def numpySSIM(img1, img2):
    # Calculate SSIM
    ssim = SSIM(img1, img2, data_range=1.0, channel_axis=2, win_size=5)
    ssim = torch.tensor(ssim)

    return ssim