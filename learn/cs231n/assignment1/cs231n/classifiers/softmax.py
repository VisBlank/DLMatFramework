import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # The non vectorized version will be slow
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  # For each example on training
  for i in xrange(num_train):
    scores = X[i].dot(W)
    # Solve numerical instability
    shift_scores = scores - max(scores)
    loss_i = -shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
    loss += loss_i

    for j in xrange(num_classes):
        softmax_output = np.exp(shift_scores[j])/sum(np.exp(shift_scores))
        if j == y[i]:
            dW[:,j] += (-1 + softmax_output) *X[i]
        else:
            dW[:,j] += softmax_output *X[i]


  # Right now the loss is a sum over all training examples, but we want it to be an average
  loss /= num_train
  # Complete loss
  loss +=  0.5* reg * np.sum(W * W)

  # Get gradient
  dW = dW/num_train + reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
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
  num_classes = W.shape[1]
  num_train = X.shape[0]

  # Get scores (vectorized), by the way this is the linear classifer with bias trick
  scores = X.dot(W)

  # Calculate the loss but first solve numerical instability
  shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)

  # Get probabilities
  probabilities = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)

  # Calculate complete loss with regularization
  loss = -np.sum(np.log(probabilities[range(num_train), list(y)]))
  loss /= num_train
  loss +=  0.5 * reg * np.sum(W * W)

  # Get the gradient dW
  # first compute dS: the gradient of the loss function with respect to the scores
  dS= probabilities.copy()
  dS[range(num_train), list(y)] += -1
  dW = (X.T).dot(dS)
  dW = dW/num_train + reg* W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

