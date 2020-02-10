import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.pool_size = 2
        self.conv_relu_conv_relu_pool = nn.Sequential(
            nn.Conv2d(im_size[0], hidden_dim, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool_size)
        )
        self.fully_connected = nn.Linear(hidden_dim * (im_size[1] // self.pool_size) * (im_size[2] // self.pool_size), n_classes)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass.
        #############################################################################
        out = self.conv_relu_conv_relu_pool(images)
        print(out.shape)
        out = self.conv_relu_conv_relu_pool(out.reshape(images.shape))
        scores = self.fully_connected(out.view(images.shape[0], -1))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores
