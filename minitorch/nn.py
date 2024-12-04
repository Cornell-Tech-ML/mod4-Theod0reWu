from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height, new_width = height // kh, width // kw
    t = input.contiguous().view(batch, channel, new_height, kh,  new_width, kw)
    return t.permute(0, 1, 2, 4, 3, 5).contiguous().view(batch, channel, new_height, new_width, kw * kh), new_height, new_width


# TODO: Implement for Task 4.3.

def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """"Computes average pooling given an input tensor and a kernel size"""
    batch, channel, height, width = input.shape
    new_tensor, a, b = tile(input, kernel)  
    return new_tensor.mean(dim=4).view(batch, channel, a, b)

# For max see tensor.py, tensor_ops.py and tensor_functions.py

def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """"Computes max pooling given an input tensor and a kernel size"""
    batch, channel, height, width = input.shape
    new_tensor, a, b = tile(input, kernel)  
    return new_tensor.max(dim=4).view(batch, channel, a, b)

def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout for a given probability"""
    if (ignore):
        return input
    return input * (rand(input.shape, input.backend) > p)

def softmax(input: Tensor, dim = None) -> Tensor:
    """Applies the softmax to the input tensor along the given dimension"""
    exp = input.exp()
    return exp / exp.sum(dim)

def logsoftmax(input: Tensor, dim = None) -> Tensor:
    """Applies the log of the softmax to the input tensor along the given dimension"""
    exp = input.exp()
    return (exp / exp.sum(dim)).log()