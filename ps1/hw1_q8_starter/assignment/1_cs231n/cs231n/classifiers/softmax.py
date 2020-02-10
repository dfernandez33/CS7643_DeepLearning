import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs:
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W, an array of same size as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = W.dot(X)
    scores -= np.max(scores, axis=0)
    num_examples = X.shape[1]

    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=0)
    p = exp_scores / sum_exp_scores
    loss = (-1/num_examples)*np.sum(np.log(p[y, np.arange(num_examples)]))
    # add the regularization term to the loss
    loss += reg * np.square(np.linalg.norm(W))

    p[y, np.arange(num_examples)] -= 1
    dW = p.dot(X.T) / num_examples
    # add regularization term to dw
    dW += 2 * reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
