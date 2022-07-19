from builtins import range
import numpy as np
from typing import Any, Dict, List


def affine_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x_rows = x.reshape((x.shape[0], -1))
    out = np.dot(x_rows, w) + b
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout: np.ndarray, cache: List[np.ndarray]):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = x.shape[0]
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(np.reshape(x, (N, -1)).T, dout)       # a^(l-1).T · (a^l-y)
    db = np.dot(np.ones(shape=(1, N)), dout)          # 1 · (a^l-y)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x: np.ndarray):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout: np.ndarray, cache: np.ndarray):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    relu_derivatives = np.zeros(shape=x.shape)
    relu_derivatives[x>=0] = 1
    dx = relu_derivatives * dout
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x: np.ndarray, y: np.ndarray):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C = x.shape
    output = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    if y is None:
      return output
    y_vec = np.zeros(shape=(N, C))
    # Naive implement
    loss = np.zeros((N))
    for i in range(N):
      loss[i] = np.log(output[i, y[i]])
    loss = -np.sum(loss) / N
    dx = output
    for i in range(N):
      dx[i, y[i]] -= 1
    dx /= N
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def conv_forward_naive(x: np.ndarray, w: np.ndarray, b: np.ndarray, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x_padded = np.pad(x, ((0, 0), (0, 0), (conv_param['pad'], conv_param['pad']), (conv_param['pad'], conv_param['pad'])))
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_prime = int(1 + (H + 2*conv_param['pad'] - HH) / conv_param['stride'])
    W_prime = int(1 + (W + 2*conv_param['pad'] - WW) / conv_param['stride'])

    out = np.zeros(shape=(N, F, H_prime, W_prime))
    for n in range(N):
      for f in range(F):
        for h in range(H_prime):
          for w_idx in range(W_prime):
            for c in range(C):
              out[n, f, h, w_idx] += np.sum(np.multiply(x_padded[n, c, h*conv_param['stride']:h*conv_param['stride']+HH, w_idx*conv_param['stride']:w_idx*conv_param['stride']+WW], w[f, c, :, :]))
            out[n, f, h, w_idx] += b[f]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout: np.ndarray, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives [N, F, H_t, W_t].
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = cache[0].shape
    F, _, HH, WW = cache[1].shape

    x_padded = np.pad(cache[0], ((0, 0), (0, 0), (cache[3]['pad'], cache[3]['pad']), (cache[3]['pad'], cache[3]['pad'])))

    dx = np.zeros_like(cache[0])
    pad_height = int(((x_padded.shape[2]-1)*cache[3]['stride']+HH-dout.shape[2])/2)
    pad_width = int(((x_padded.shape[3]-1)*cache[3]['stride']+WW-dout.shape[3])/2)
    dout_padded = np.pad(dout, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)))
    w_rot = np.flip(cache[1], axis=(2, 3))
    for n in range(N):
      for c in range(C):
        for h in range(H):
          for w in range(W):
            for f in range(F):
              dx[n, c, h, w] += np.sum(np.multiply(dout_padded[n, f, h*cache[3]['stride']:h*cache[3]['stride']+HH, w*cache[3]['stride']:w*cache[3]['stride']+WW], w_rot[f, c, :, :]))

    dw = np.zeros_like(cache[1])
    for f in range(F):
      for c in range(C):
        for h in range(HH):
          for w in range(WW):
            for n in range(N):
              dw[f, c, h, w] += np.sum(np.multiply(x_padded[n, c, h*cache[3]['stride']:h*cache[3]['stride']+dout.shape[2], w*cache[3]['stride']:w*cache[3]['stride']+dout.shape[3]], 
                                            dout[n, f, :, :]))

    db = np.zeros_like(cache[2])
    db = np.sum(dout.swapaxes(1, 0), axis=(1, 2, 3))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x: np.ndarray, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    stride = pool_param['stride']
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    out = np.zeros((N, C, int(1 + (H - pool_height) / stride), int(1 + (H - pool_height) / stride)))
    for n in range(N):
      for c in range(C):
        for h in range(out.shape[2]):
          for w in range(out.shape[3]):
            out[n, c, h, w] = np.max(x[n, c, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout: np.ndarray, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = cache[0].shape
    stride = cache[1]['stride']
    pool_height = cache[1]['pool_height']
    pool_width = cache[1]['pool_width']
    dx = np.zeros_like(cache[0])
    for n in range(N):
      for c in range(C):
        for h in range(dout.shape[2]):
          for w in range(dout.shape[3]):
            relative_x, relative_y = np.unravel_index(np.argmax(cache[0][n, c, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width]), (pool_height, pool_width))
            dx[n, c, relative_x+h*stride, relative_y+w*stride] = dout[n, c, h, w]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
