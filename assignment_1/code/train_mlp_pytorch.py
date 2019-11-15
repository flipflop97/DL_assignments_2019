"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    targets: 2D int array of size [batch_size, n_classes]
             with one-hot encoding. Ground truth labels for
             each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  accuracy = torch.mean((predictions.argmax(axis=-1) == targets.argmax(axis=-1)).float())
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope
  
  ########################
  # PUT YOUR CODE HERE  #
  #######################
  data = cifar10_utils.get_cifar10(FLAGS.data_dir)

  n_inputs = np.prod(data['train'].images.shape[1:])
  n_classes = data['train'].labels.shape[1]

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  mlp = MLP(n_inputs, dnn_hidden_units, n_classes, neg_slope).to(device)
  cel = torch.nn.CrossEntropyLoss().to(device)

  params = mlp.parameters()
  optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate)

  test_input = data['test'].images.reshape(data['test'].num_examples, -1)
  test_target = data['test'].labels

  losses = torch.zeros(FLAGS.max_steps, device=device)
  accuracies = torch.zeros(FLAGS.max_steps, device=device)

  for step in range(FLAGS.max_steps):
    optimizer.zero_grad()

    batch, train_target = data['train'].next_batch(FLAGS.batch_size)
    train_input = batch.reshape(FLAGS.batch_size, -1)

    train_input = torch.tensor(train_input, device=device)
    train_target = torch.tensor(train_target, device=device)

    out = mlp.forward(train_input)
    loss = cel.forward(out, train_target.argmax(axis=-1))

    loss.backward()
    optimizer.step()

    losses[step] = loss.detach()
    accuracies[step] = accuracy(out, train_target)
  
  from matplotlib import pyplot as plt

  plt.plot(losses.cpu())
  plt.title('PyTorch')
  plt.xlabel('step')
  plt.ylabel('loss')
  plt.show()
  
  plt.plot(accuracies.cpu())
  plt.title('PyTorch')
  plt.xlabel('step')
  plt.ylabel('accuracy')
  plt.show()
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  FLAGS, unparsed = parser.parse_known_args()

  main()