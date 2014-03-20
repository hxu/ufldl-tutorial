from __future__ import division
import math
from matplotlib.colors import Normalize
import numpy as np
import scipy.io
from matplotlib import pyplot


N_INPUT = 8 * 8
N_HIDDEN = 25
SPARSITY = 0.01
LMBDA = 0.0001
BETA = 3


def get_raw_images():
    mat = scipy.io.loadmat('IMAGES.mat')
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


def show_network(images, side_length, sample=1):
    """
    Expects images to be (n_pixels, n_images).  n_pixels should be == side_length ** 2
    """
    n_images = int(images.shape[1] * sample)
    cols = int(math.sqrt(n_images))
    rows = math.ceil(n_images / cols)
    # padding
    image_size = side_length + 1
    output = np.ones((image_size * rows, image_size * cols))
    image_mask = np.random.randint(0, images.shape[1], n_images)
    norm = Normalize()
    for i, img in enumerate(image_mask):
        this_image = images[:, img].reshape((side_length, side_length)).copy()
        # Center and normalize
        this_image -= this_image.mean()
        this_image = norm(this_image)
        # Get offsets
        offset_col = image_size * (i % cols)
        offset_col_end = offset_col + image_size - 1
        offset_row = image_size * (math.floor(i / cols))
        offset_row_end = offset_row + image_size - 1
        output[offset_row:offset_row_end, offset_col:offset_col_end] = this_image

    pyplot.imshow(output)
    pyplot.show()


def initialize_parameters(n_hidden, n_input):
    r = math.sqrt(6) / math.sqrt(n_hidden + n_input + 1)
    # Not sure if this is the right random number generator
    W1 = np.random.normal(0, 0.01 ** 2, n_hidden * n_input).reshape((n_hidden, n_input)) * 2 * r - r
    W2 = np.random.normal(0, 0.01 ** 2, n_hidden * n_input).reshape((n_input, n_hidden)) * 2 * r - r

    b1 = np.zeros((n_hidden, 1))
    b2 = np.zeros((n_input, 1))

    return W1, W2, b1, b2


def unroll(W1, W2, b1, b2):
    # theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
    pass


def sparse_autoencoder_cost(data, W1, W2, b1, b2, n_hidden, n_input, lmbda=0, sparsity=0, beta=0):
    pass
