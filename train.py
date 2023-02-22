import model
import data
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
    k_grad_fn = jax.grad(graph_fn, argnums=[1,2,3,4,5,6,7,8])
    k_grads = k_grad_fn(image, mask, *model.kernels)
    for i in range(len(model.kernels)):
        model.kernels[i] -= 0.1 * k_grads[i]

from_path = sys.argv[1]
to_path = sys.argv[2]
model.load_kernels(from_path)

train_steps = 1 if sys.argv[3] == '' else int(sys.argv[3])
print("training for {} steps".format(train_steps))

for i in range(1, train_steps+1):
    try:
        image, mask = data.load_random_image_chunk('sample_image_set', 8, data.cocotext)
        mask = np.expand_dims(mask, axis=-1)
        #print(mask.shape)
        print("step: {}".format(i))
        print(graph_fn(image, mask, *model.kernels))
        step_start = time.time()
        optimize2(image, mask)
        print("step finished after {}".format(time.time() - step_start))
        print(graph_fn(image, mask, *model.kernels))
    except:
        print("training step failed")

model.save_kernels(to_path)
print("saved model weights")