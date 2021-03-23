import numpy as np

def mask(corruption_level ,size):
    mask = np.random.binomial(1, 1 - corruption_level, [size[0],size[1]])
    return mask

def add_noise(x , corruption_level ):
    x = x * mask(corruption_level , x.shape)
    return x
