import numpy as np

from layers import *
from layer_utils import *


class mnist(object):
  """
  LeNet in caffe
  conv - 2x2 max pool - conv - 2x2 max pool - affine - relu - affine - softmax

  """
  
  def __init__(self, input_dim=(1,28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):

    self.params = {}
    self.reg = reg
    self.dtype = dtype

    # xavier initialization
    self.params['W1'] = np.random.randn(20,1,5,5) / np.sqrt(20)
    self.params['b1'] = np.zeros((20,))
    self.params['W2'] = np.random.randn(50,20,5,5) / np.sqrt(50)
    self.params['b2'] = np.zeros((50,))
    self.params['W3'] = np.random.randn(800,500) / np.sqrt(800)
    self.params['b3'] = np.zeros((500,))
    self.params['W4'] = np.random.randn(500,10) / np.sqrt(500)
    self.params['b4'] = np.zeros((10,))

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': 0}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    cnn1_out, cnn1_cache = conv_pool_forward(X, W1, b1, conv_param, pool_param)
    cnn2_out, cnn2_cache = conv_pool_forward(cnn1_out, W2, b2, conv_param, pool_param)
    af1_out, af1_cache = affine_relu_forward(cnn2_out, W3, b3)
    scores, af2_cache = affine_forward(af1_out, W4, b4)
    
    if y is None:
      return scores
    
    loss, grads = 0, {}

    data_loss,dscores = softmax_loss(scores,y)
    daf1_out,dW4,db4 = affine_backward(dscores,af2_cache)
    dcnn2_out,dW3,db3 = affine_relu_backward(daf1_out,af1_cache)
    dcnn1_out,dW2,db2 = conv_pool_backward(dcnn2_out,cnn2_cache)
    dX, dW1, db1 = conv_pool_backward(dcnn1_out, cnn1_cache)
    # print self.reg
    grads['W1'] = dW1 + self.reg * W1
    grads['W2'] = dW2 + self.reg * W2
    grads['W3'] = dW3 + self.reg * W3
    grads['W4'] = dW4 + self.reg * W4
    grads['b1'] = db1 * 2 # in caffe the lr_mult = 2
    grads['b2'] = db2 * 2
    grads['b3'] = db3 * 2
    grads['b4'] = db4 * 2

    reg_loss = 0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3)+np.sum(W4*W4))
    loss = data_loss + reg_loss
    
    return loss, grads
  
  
pass
