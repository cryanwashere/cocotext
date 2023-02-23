import model
import outdated_stuff_that_I_dont_want_to_delete.data as data
import jax
import sys
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

model = model.ConvModel()
def softmax(x):
    x = jnp.exp(x)
    x = x / jnp.sum(x)
    return x

def binary_crossentropy(x,y):
    smx = softmax(x)
    return -jnp.mean(y * jnp.log(smx))

def graph_fn(x, y, *kernels):
    out = model.model_fn(x, *kernels)
    loss = binary_crossentropy(out, y)
    return loss

def graph_fn2(x, y, *kernels):
    out = model.model_fn(x, *kernels)
    loss = binary_crossentropy(out, y)
    return loss, out


def optimize(image, mask):
    for i, kernel in enumerate(model.kernels):
        k_grad_fn = jax.grad(graph_fn, argnums=i+2)
        k_grad = k_grad_fn(image, mask, *model.kernels)
        #print("found kernel gradients, shape: {}".format(k_grad.shape))
        model.kernels[i] -= 0.01 * k_grad
def optimize2(image, mask):
    
    autodif_start = time.time()
    k_grad_fn = jax.grad(graph_fn, argnums=[2,3,4,5,6,7,8,9])
    print("autodif complete after: {}".format(time.time()-autodif_start))

    grad_start = time.time()
    k_grads = k_grad_fn(image, mask, *model.kernels)
    print("grad complete after: {}".format(time.time()-grad_start))

    for i in range(len(model.kernels)):
        model.kernels[i] -= 0.1 * k_grads[i]

