import numpy as np
from module import Module


class SoftmaxCrossentropy(Module):
    def __init__(self, name):
        super(SoftmaxCrossentropy, self).__init__(name)

    def forward(self, x, **kwargs):
        y = kwargs.pop('y', None)
        """
        x: input array.
        y: real labels for this input.
        probs: probabilities of labels for this input.
        loss: cross entropy loss between probs and real labels.
        **Save whatever you need for backward pass in self.cache.
        """
        probs = None
        loss = None
        # todo: implement the forward propagation for probs and compute cross entropy loss
        # NOTE: implement a numerically stable version.If you are not careful here
        # it is easy to run into numeric instability!
   
        return loss, probs

    def backward(self, dout=0):
        dx = None
        # todo: implement the backward propagation for this layer.

        return dx
