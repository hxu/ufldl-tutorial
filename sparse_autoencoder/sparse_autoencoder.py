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


def initialize_parameters(n_input, n_hidden):
    r = math.sqrt(6) / math.sqrt(n_hidden + n_input + 1)
    # Not sure if this is the right random number generator
    W1 = np.random.random(n_hidden * n_input).reshape((n_hidden, n_input)) * 2 * r - r
    W2 = np.random.random(n_hidden * n_input).reshape((n_input, n_hidden)) * 2 * r - r

    b1 = np.zeros((n_hidden, 1))
    b2 = np.zeros((n_input, 1))

    return W1, W2, b1, b2


def unroll(W1, W2, b1, b2):
    # theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
    pass


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_p(x):
    return np.multiply(x, (1 - x))


def autoencoder_single_pass(X, W1, W2, b1, b2, n_hidden, n_input, lmbda=0, sparsity=0, beta=0):
    """
    Single hidden layer sparse autoencoder

    In other words, the input layer has 64 units, the hidden layer has 25 hidden units, and the output layaer
    again has 64 hidden units.  The autoencoder is trying to reconstruct the original input units

    Arguments:
    ==========
    X: ndarray, shape (n_input, n_obs)

    W1: ndarray, shape (n_hidden, n_input)

    W2: ndarray, shape (n_input, n_hidden)

    b1: ndarray, shape (n_hidden, 1)

    b2: ndarray, shape (n_input, 1)
    """
    # Forward propagation

    # Z2 is the inputs into the second layer
    # (n_hidden, n_obs) = (n_hidden, n_obs) + (n_hidden, 1), b1 will project
    Z2 = (W1 * X) + b1
    # A2 is the activation from the second layer
    # (n_hidden, n_obs)
    A2 = sigmoid(Z2)
    # Z3 is inputs into the third layer
    # (n_input, n_obs)
    Z3 = (W2 * A2) + b2

    # y_pred is the same shape as in put data
    # (n_input, n_obs)
    y_pred = sigmoid(Z3)
    cost = cost_func(X, y_pred, W1, W2, lmbda)

    # Back propagation
    # initialize gradients
    gradW1 = np.zeros(W1.shape)
    gradW2 = np.zeros(W2.shape)

    # d3 is the error in the third layer
    # (n_input, n_obs) = (n_input, n_obs) .* (n_input, n_obs)
    d3 = np.multiply(-(X - y_pred), sigmoid_p(Z3))
    # (n_hidden, n_obs) = (n_hidden, n_input) * (n_input, n_obs) .* (n_hidden, n_obs)
    d2 = np.multiply(W2.T * d3, sigmoid_p(Z2))

    # (n_input, n_hidden) = (n_input, n_obs) * (n_obs, n_hidden)
    gradW2 = gradW2 + d3 * A2.T
    gradW1 = gradW1 + d2 * X.T
    gradb2 = d3
    gradb1 = d2

    return cost, (gradW1, gradW2, gradb1, gradb2)


def cost_func(y, y_pred, W1, W2, lmbda=0):
    """
    Assumes y is in the shape of (n_preds, n_obs)
    """
    assert y.shape == y.shape, "Ndarrays are not of the same shape: {} and {}".format(y.shape, y.shape)
    err = np.sum((y_pred - y) ** 2) / y.shape[1]
    # Weight decay
    reg = (lmbda / 2) * np.sum(W1 ** 2) + np.sum(W2 ** 2)
    return err + reg


def compute_numerical_gradient(func, params):
    # General idea seems to be that we want to perturb every single element of each parameter by a small
    # amount, subtract from the original, then check against
    #
    # Expects func to return in the form of value, (gradient elements)
    pert = 1e-4
    outputs = [np.zeros(p.shape, dtype=np.float64) for p in params]
    perturbs = [np.zeros(p.shape, dtype=np.float64) for p in params]
    for ip, p in enumerate(params):
        for ie, e in enumerate(np.nditer(p)):
            perturbs[ip].flat[ie] = pert
            args = [a + perturbs[ia] for ia, a in enumerate(params)]
            val1, grad1 = func(*args)

            args = [a - perturbs[ia] for ia, a in enumerate(params)]
            val2, grad2 = func(*args)
            outputs[ip].flat[ie] = (val1 - val2) / (2 * pert)
            perturbs[ip].flat[ie] = 0

    return outputs


def check_gradient():
    x = np.array([4, 10])
    val, grad = simple_quadratic(x)
    grad = np.concatenate([xx.flatten() for xx in grad])
    numerical_grad = compute_numerical_gradient(simple_quadratic, [x])
    numerical_grad = np.concatenate([xx.flatten() for xx in numerical_grad])
    print "Gradient"
    print grad
    print "Numerical gradient"
    print numerical_grad

    diff = np.linalg.norm(numerical_grad - grad) / np.linalg.norm(numerical_grad + grad)
    print "Norm of differences.  Should be less than 2.1452e-12"
    print diff


def simple_quadratic(x):
    val = (x[0] ** 2) + (3 * x[0] * x[1])
    grad = np.zeros((2, ))
    grad[0] = (2 * x[0]) + (3 * x[1])
    grad[1] = 3 * x[0]
    return val, (grad, )
