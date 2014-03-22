from __future__ import division
import math
from matplotlib.colors import Normalize
import numpy as np
import scipy.io
from matplotlib import pyplot, cm
from operator import mul
from scipy.optimize import minimize


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
    output = np.zeros((image_size * rows, image_size * cols))
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
    pyplot.set_cmap(cm.gray)
    pyplot.show()


def initialize_parameters(n_input, n_hidden):
    r = math.sqrt(6) / math.sqrt(n_hidden + n_input + 1)
    # Not sure if this is the right random number generator
    W1 = np.random.random(n_hidden * n_input).reshape((n_hidden, n_input)) * 2 * r - r
    W2 = np.random.random(n_hidden * n_input).reshape((n_input, n_hidden)) * 2 * r - r

    b1 = np.zeros((n_hidden, 1))
    b2 = np.zeros((n_input, 1))

    return W1, W2, b1, b2


class Unroller(object):
    """
    Unrolls an array of ndarrays into a single ndarray.
    Stores the shapes so that the ndarray can be re-rolled into the array of ndarrays
    """
    def __init__(self, *args):
        self.shapes = [xx.shape for xx in args]
        self.args = args
        self.total_elems = sum([xx.size for xx in self.args])

    def flatten(self, *args):
        if not args:
            return np.concatenate([xx.flatten() for xx in self.args])
        else:
            return np.concatenate([xx.flatten() for xx in args])

    def roll(self, arg):
        assert arg.size == self.total_elems, "Expected array of {} elements".format(self.total_elems)
        pointer = 0
        res = []
        for s in self.shapes:
            count = reduce(mul, s)
            this_res = arg[pointer:pointer+count].reshape(s)
            res.append(this_res)
            pointer = pointer + count

        return res


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_p(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def autoencoder_single_pass(X, W1, W2, b1, b2, lmbda=0, sparsity=0, beta=0):
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
    n_obs = X.shape[1]

    # Z2 is the inputs into the second layer
    # (n_hidden, n_obs) = (n_hidden, n_obs) + (n_hidden, 1), b1 will project
    Z2 = np.dot(W1, X) + b1
    # A2 is the activation from the second layer
    # (n_hidden, n_obs)
    A2 = sigmoid(Z2)
    # Z3 is inputs into the third layer
    # (n_input, n_obs)
    Z3 = np.dot(W2, A2) + b2

    rho_hat = A2.sum(1) / n_obs

    # y_pred is the same shape as in put data
    # (n_input, n_obs)
    y_pred = sigmoid(Z3)
    sparsity_penalty = beta * kl(sparsity, rho_hat)
    cost = cost_func(X, y_pred, W1, W2, lmbda) + sparsity_penalty

    # Back propagation
    # initialize gradients
    gradW1 = np.zeros(W1.shape)
    gradW2 = np.zeros(W2.shape)

    # d3 is the error in the third layer
    # (n_input, n_obs) = (n_input, n_obs) .* (n_input, n_obs)
    d3 = (y_pred - X) * sigmoid_p(Z3)
    #
    d2_penalty = kl_delta(sparsity, rho_hat).reshape(rho_hat.shape[0], 1)
    # (n_hidden, n_obs) = (n_hidden, n_input) * (n_input, n_obs) .* (n_hidden, n_obs)
    d2 = (np.dot(W2.T, d3) + beta * d2_penalty) * sigmoid_p(Z2)

    # (n_input, n_hidden) = (n_input, n_obs) * (n_obs, n_hidden)
    gradW2 = lmbda * W2 + np.dot(d3, A2.T) / n_obs
    gradW1 = lmbda * W1 + np.dot(d2, X.T) / n_obs
    gradb2 = d3.sum(1).reshape(b2.shape) / n_obs
    gradb1 = d2.sum(1).reshape(b1.shape) / n_obs

    return cost, (gradW1, gradW2, gradb1, gradb2)


def cost_func(y, y_pred, W1, W2, lmbda=0):
    """
    Assumes y is in the shape of (n_preds, n_obs)
    """
    assert y.shape == y.shape, "Ndarrays are not of the same shape: {} and {}".format(y.shape, y.shape)
    err = np.sum((y_pred - y) ** 2) / (2 * y.shape[1])
    # Weight decay
    reg = (lmbda / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    # print "Err: {}, Reg: {}".format(err, reg)
    return err + reg


def kl(r, rh):
    if r == 0:
        return np.zeros(rh.shape)
    else:
        return np.sum(r * np.log(r / rh) + (1 - r) * np.log((1-r) / (1 - rh)))


def kl_delta(r, rh):
    if r == 0:
        return np.zeros(rh.shape)
    else:
        return -(r / rh) + (1 - r) / (1 - rh)


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


def check_gradient2():
    patches = generate_patches(get_raw_images())[:, 0].reshape((64, 1))
    W1, W2, b1, b2 = initialize_parameters(N_INPUT, N_HIDDEN)
    cost, grad = autoencoder_single_pass(patches, W1, W2, b1, b2, LMBDA, SPARSITY, BETA)

    def wrapper(W1, W2, b1, b2):
        return autoencoder_single_pass(patches, W1, W2, b1, b2, LMBDA, SPARSITY, BETA)

    num_grad = compute_numerical_gradient(wrapper, [W1, W2, b1, b2])

    grad = np.concatenate([xx.flatten() for xx in grad])
    num_grad = np.concatenate([xx.flatten() for xx in num_grad])
    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
    print "Norm of differences.  Should be less than 1e-9"
    print diff


def simple_quadratic(x):
    val = (x[0] ** 2) + (3 * x[0] * x[1])
    grad = np.zeros((2, ))
    grad[0] = (2 * x[0]) + (3 * x[1])
    grad[1] = 3 * x[0]
    return val, (grad, )


def train():
    patches = generate_patches(get_raw_images())
    W1, W2, b1, b2 = initialize_parameters(N_INPUT, N_HIDDEN)
    unroller = Unroller(W1, W2, b1, b2)

    def obj_func(theta):
        W1, W2, b1, b2 = unroller.roll(theta)
        cost, grad = autoencoder_single_pass(patches, W1, W2, b1, b2, LMBDA, SPARSITY, BETA)
        return cost, unroller.flatten(grad)

    res = minimize(obj_func, unroller.flatten(), method='BFGS', jac=True, options={'maxiter': 400, 'disp':True})

    W1, W2, b1, b2 = unroller.roll(res['x'])
