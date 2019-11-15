"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
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
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  data = cifar10_utils.get_cifar10(FLAGS.data_dir)

  n_channels = np.prod(data['train'].images.shape[1])
  n_classes = data['train'].labels.shape[1]

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  cnn = ConvNet(n_channels, n_classes).to(device)
  cel = torch.nn.CrossEntropyLoss().to(device)

  params = cnn.parameters()
  optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate)

  test_input = data['test'].images
  test_target = data['test'].labels

  test_input = torch.tensor(test_input, device=device)
  test_target = torch.tensor(test_target, device=device)

  train_losses = torch.zeros(FLAGS.max_steps, device=device)
  train_accuracies = torch.zeros(FLAGS.max_steps, device=device)
  test_losses = torch.zeros(FLAGS.max_steps // FLAGS.eval_freq, device=device)
  test_accuracies = torch.zeros(FLAGS.max_steps // FLAGS.eval_freq, device=device)

  for step in range(FLAGS.max_steps):
    optimizer.zero_grad()

    train_input, train_target = data['train'].next_batch(FLAGS.batch_size)

    train_input = torch.tensor(train_input, device=device)
    train_target = torch.tensor(train_target, device=device)

    # Use a subset of test data as this fits on my GPU
    test_input, test_target = data['test'].next_batch(4096)
    test_input = torch.tensor(test_input, device=device)
    test_target = torch.tensor(test_target, device=device)

    train_output = cnn.forward(train_input)
    train_loss = cel.forward(train_output, train_target.argmax(axis=-1))
    train_accuracy = accuracy(train_output, train_target)

    if step % FLAGS.eval_freq == 0:
      with torch.no_grad():
        test_output = cnn.forward(test_input)
        test_loss = cel.forward(test_output, test_target.argmax(axis=-1))
        test_accuracy = accuracy(test_output, test_target)
    
        print('{}\t{}\t{}'.format(step, test_loss.item(), test_accuracy.item()))

    train_losses[step] = train_loss.detach()
    train_accuracies[step] = train_accuracy
    test_losses[step // FLAGS.eval_freq] = test_loss.detach()
    test_accuracies[step // FLAGS.eval_freq] = test_accuracy

    train_loss.backward()
    optimizer.step()
  
  from matplotlib import pyplot as plt

  test_steps = torch.range(0, FLAGS.max_steps-1, FLAGS.eval_freq)

  plt.plot(train_losses.cpu(), label='Train')
  plt.plot(test_steps, test_losses.cpu(), label='Test')
  plt.title('PyTorch CNN')
  plt.xlabel('Step')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig('../report/pytorch_cnn_loss.svg', format='svg')

  plt.close()
  
  plt.plot(train_accuracies.cpu(), label='Train')
  plt.plot(test_steps, test_accuracies.cpu(), label='Test')
  plt.title('PyTorch CNN')
  plt.xlabel('Step')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.savefig('../report/pytorch_cnn_accuracy.svg', format='svg')
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
  FLAGS, unparsed = parser.parse_known_args()

  main()