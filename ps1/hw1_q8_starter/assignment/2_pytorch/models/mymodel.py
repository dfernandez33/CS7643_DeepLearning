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
        self.height, self.width = im_size[1], im_size[2]
        self.num_filters = hidden_dim
        self.num_classes = n_classes
        padding_size = (kernel_size - 1) // 2
        self.conv_relu_x4_pool_1 = nn.Sequential(
            nn.Conv2d(im_size[0], hidden_dim, kernel_size=kernel_size, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(.05),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.pool_size)
        )
        self.conv_relu_x4_pool_2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(.05),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.pool_size)
        )
        self.conv_relu_x4_pool_3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(.05),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=padding_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.pool_size)
        )
        flattened_conv_size = self.num_filters * (self.height // (self.pool_size**3)) * (self.width // (self.pool_size**3))
        self.fully_connected_block = nn.Sequential(
            nn.Linear(flattened_conv_size, flattened_conv_size),
            nn.Linear(flattened_conv_size, flattened_conv_size),
            nn.Linear(flattened_conv_size, n_classes)
        )
        self.conv_block = nn.Sequential(
            self.conv_relu_x4_pool_1,
            self.conv_relu_x4_pool_2,
            self.conv_relu_x4_pool_3
        )
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
        out = self.conv_block(images)
        flattened_out = out.view(out.shape[0], -1)  # flatten output before first fc layer
        scores = self.fully_connected_block(flattened_out)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores