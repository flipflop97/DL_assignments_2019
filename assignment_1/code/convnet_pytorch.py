"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()

    neg_slope = 0.02

    self.conv = nn.Sequential(
      # conv1
      nn.Conv2d(3, 64, 3, 1, 1),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(neg_slope),

      # maxpool1
      nn.MaxPool2d(3, 2, 1),

      # conv2
      nn.Conv2d(64, 128, 3, 1, 1),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(neg_slope),

      # maxpool2
      nn.MaxPool2d(3, 2, 1),

      # conv3_a
      nn.Conv2d(128, 256, 3, 1, 1),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(neg_slope),

      # conv3_b
      nn.Conv2d(256, 256, 3, 1, 1),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(neg_slope),

      # maxpool3
      nn.MaxPool2d(3, 2, 1),

      # conv4_a
      nn.Conv2d(256, 512, 3, 1, 1),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(neg_slope),

      # conv4_b
      nn.Conv2d(512, 512, 3, 1, 1),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(neg_slope),

      # maxpool4
      nn.MaxPool2d(3, 2, 1),

      # conv5_a
      nn.Conv2d(512, 512, 3, 1, 1),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(neg_slope),

      # conv5_b
      nn.Conv2d(512, 512, 3, 1, 1),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(neg_slope),

      # maxpool5
      nn.MaxPool2d(3, 2, 1)
    )

    # linear
    self.linear = nn.Linear(512, n_classes)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    conv = self.conv.forward(x)
    out = self.linear(conv.view(conv.size(axis=0), -1))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
