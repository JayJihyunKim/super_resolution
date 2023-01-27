import math
import random

import matplotlib.pyplot as plt
import numpy as np
from cv2 import warpPerspective, INTER_CUBIC
from math import pi, cos, sin, floor
import cv2

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def random_agument(im, crop_size=256, max_scale=1.0, min_scale=0.5, shear_sigma=0.01):

    # shifting to center
    shift_to_center_mat = np.array([[1,0,-im.width/2],
                                   [0,1,-im.height/2],
                                   [0,0,1]])
    # shift back from center
    shift_back_from_center = np.array([[1,0,im.width/2],
                                   [0,1,im.height/2],
                                   [0,0,1]])
    # scaling
    s_x = np.clip(random.uniform(floor(min_scale), max_scale),min_scale,max_scale)
    s_y = np.clip(random.uniform(floor(min_scale), max_scale),min_scale,max_scale)
    scale_mat = np.array([[s_x,0,0],
                         [0,s_y,0],
                         [0,0,1]])
    # rotation
    theta = np.random.rand()*pi*2
    rotation_mat = np.array([[cos(theta),-sin(theta),0],
                            [sin(theta),cos(theta),0],
                            [0,0,1]])

    # shearing
    sh_x = np.random.randn()*shear_sigma
    sh_y = np.random.randn()*shear_sigma
    shear_mat = np.array([[1,sh_x,0],
                         [sh_y,1,0],
                         [0,0,1]])
    # shift
    shift_x = np.random.rand() * np.clip(s_x * im.width - crop_size, 0, 9999)
    shift_y = np.random.rand() * np.clip(s_y * im.height - crop_size, 0, 9999)
    # shift_x = np.random.randn()
    # shift_y = np.random.randn()
    shift_mat = np.array([[1, 0, - shift_x],
                          [0, 1, - shift_y],
                          [0, 0, 1]])

    transform_mat = (shift_back_from_center
                     .dot(shift_mat)
                     .dot(shear_mat)
                     .dot(rotation_mat)
                     .dot(scale_mat)
                     .dot(shift_to_center_mat))

    im = np.asarray(im)
    im = im/255.

    return np.clip(warpPerspective(im, transform_mat, (crop_size, crop_size), flags=INTER_CUBIC), 0, 1)

def psnr(img1, img2):
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return 100
    if np.max(img1) <= 1.0:
        PIXEL_MAX = 1.0
    else:
        PIXEL_MAX = 255.0

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianLernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1*mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1*img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return ssim_map.mean()
