from __future__ import division
import math
import numpy as np
import scipy as sp


def get_raw_images():
    mat = sp.io.loadmat('IMAGES.mat')
    # Shape (512, 512, 10) - 10 images of 512 by 512
    return mat['IMAGES']


def generate_patches(images, n_patches=10000, patch_size=8):
    """
    Expect images to be an ndarray of (512, 512, n_images) shape
    Returns ndarray of (patch_size ** 2, n_patches)
    """
    n_images = images.shape[2]
    img_size = images.shape[0]
    res = np.zeros((patch_size ** 2, n_patches))
    for i in xrange(n_patches):
        img_idx = i % n_images
        offset_col = np.random.randint(img_size - patch_size + 1)
        offset_row = np.random.randint(img_size - patch_size + 1)
        patch = images[offset_row:offset_row + patch_size, offset_col:offset_col + patch_size, img_idx]
        res[:, i] = patch.flatten()

    return res


def show_network(images, side_length):
    """
    Expects images to be (n_pixels, n_images).  n_pixels should be == side_length ** 2
    """
    n_images = images.shape[1]
    cols = int(math.sqrt(n_images))
    rows = math.ceil(n_images / cols)
    # padding
    image_size = side_length + 1
    output = np.ones((image_size * rows, image_size * cols))