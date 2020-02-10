import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
    We multiply this against a weight matrix of shape (D, M) where
    D = \prod_i d_i

    Inputs:
    x - Input data, of shape (N, d_1, ..., d_k)
    w - Weights, of shape (D, M)
    b - Biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    X = x.reshape(x.shape[0], np.prod(x[0].shape))
    out = X.dot(w) + b
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    X = x.reshape(x.shape[0], np.prod(x[0].shape))
    dw = (dout.T.dot(X)).T
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    db = np.sum(dout, axis=0)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    out = np.maximum(np.zeros_like(x), x)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    dx = np.multiply(np.where(x <= 0, 0, 1), dout)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    stride, pad = conv_param['stride'], conv_param['pad']
    num_imgs, im_channels, height, width = x.shape
    num_filters, filter_channels, filter_height, filter_width = w.shape

    h_prime = int((height + (2 * pad) - filter_height) / stride) + 1
    w_prime = int((width + (2 * pad) - filter_width) / stride) + 1
    padded_x = np.pad(x, ((0, 0), (0, 0), (pad, pad),  (pad, pad)), 'constant')

    W_row = w.reshape(num_filters, filter_channels * filter_height * filter_width)
    X_col = np.zeros((filter_width * filter_height * filter_channels, h_prime * w_prime))
    out = np.zeros((num_imgs, num_filters, h_prime, w_prime))
    for img in range(num_imgs):
        index = 0
        for r in range(0, padded_x.shape[2] - filter_height + 1, stride):
            for c in range(0, padded_x.shape[3] - filter_width + 1, stride):
                current_col = padded_x[img, :, r:r+filter_height, c:c+filter_width].flatten()
                X_col[:, index] = current_col
                index += 1
        out[img] = (W_row.dot(X_col) + b.reshape(num_filters, 1)).reshape(num_filters, h_prime, w_prime)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    x, w, b, conv_params = cache
    stride, pad = conv_params['stride'], conv_params['pad']
    num_imgs, _, height, width = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    padded_x = np.pad(x, ((0, 0), (0, 0), (pad, pad),  (pad, pad)), 'constant')

    dx_pad = np.zeros_like(padded_x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    for img in range(num_imgs):
        for curr_filter in range(num_filters):
            db[curr_filter] += np.sum(dout[img, curr_filter])
            for r in range(0, padded_x.shape[2] - filter_height + 1, stride):
                for c in range(0, padded_x.shape[3] - filter_width + 1, stride):
                    curr_conv = padded_x[img, :, r:r+filter_height, c:c+filter_width]
                    dw[curr_filter] += curr_conv * dout[img, curr_filter, r, c]
                    dx_pad[img, :, r:r+filter_height, c:c+filter_width] += w[curr_filter] * dout[img, curr_filter, r, c]
    # take the values of dx_pad and remove the padding
    dx = dx_pad[:, :, pad:height+pad, pad:width+pad]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    num_imgs, channels, height, width = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    h_prime = int(1 + (height - pool_height) / stride)
    w_prime = int(1 + (width - pool_width) / stride)
    out = np.zeros((num_imgs, channels, h_prime, w_prime))
    for img in range(0, num_imgs):
        index = 0
        img_pools = np.zeros((channels, h_prime * w_prime))
        for r in range(0, height - pool_height + 1, stride):
            for c in range(0, width - pool_width + 1, stride):
                pool = x[img, :, r:r+pool_height, c:c+pool_width].reshape(channels, pool_width * pool_height)
                img_pools[:, index] = pool.max(axis=1)
                index += 1
        out[img] = img_pools.reshape((channels, h_prime, w_prime))
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    num_imgs, channels, h_prime, w_prime = dout.shape
    x, pool_params = cache
    pool_height, pool_width, stride = pool_params['pool_height'], pool_params['pool_width'], pool_params['stride']
    height, width = x.shape[2], x.shape[3]

    dx = np.zeros_like(x)
    for img in range(0, num_imgs):
        index = 0
        img_row_grad = dout[img].reshape((channels, h_prime * w_prime))
        for r in range(0, height - pool_height + 1, stride):
            for c in range(0, width - pool_width + 1, stride):
                # calculate max indices for the current pool
                pool = x[img, :, r:r+pool_height, c:c+pool_width].reshape(channels, pool_width * pool_height)
                max_indices = pool.argmax(axis=1)
                # apply gradient only at the max indices
                pool_grad = np.zeros_like(pool)
                pool_grad[np.arange(channels), max_indices] = img_row_grad[:, index]
                index += 1
                # update dx with the gradient of the current pool
                dx[img, :, r:r+pool_height, c:c+pool_width] += pool_grad.reshape((channels, pool_height, pool_width))
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

