import numpy as np

# some helper functions that rotate arrays on the trailing axes.
# these should work for both Theano expressions and numpy arrays.

def array_tf_0(arr):
    return arr

def array_tf_90(arr):
    axes_order = range(arr.ndim - 2) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)

def array_tf_180(arr):
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None, None, -1)]
    return arr[tuple(slices)]

def array_tf_270(arr):
    axes_order = range(arr.ndim - 2) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)


def array_tf_0f(arr): # horizontal flip
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)]

def array_tf_90f(arr):
    axes_order = range(arr.ndim - 2) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None), slice(None)]
    # slicing does nothing here, technically I could get rid of it.
    return arr[tuple(slices)].transpose(axes_order)

def array_tf_180f(arr):
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None)]
    return arr[tuple(slices)]

def array_tf_270f(arr):
    axes_order = range(arr.ndim - 2) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)