import os

import jax
import numpy as np
import jax.numpy as jnp
from jax.lax import conv_general_dilated

def ReLU(x):
    return jnp.maximum(x,0)
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def softmax(x):
    x = jnp.exp(x)
    x = x / jnp.sum(x)
    return x

'''
zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
cond = (logits >= zeros)
relu_logits = array_ops.where(cond, logits, zeros)
neg_abs_logits = array_ops.where(cond, -logits, logits)
return math_ops.add(
    relu_logits - logits * labels,
    math_ops.log1p(math_ops.exp(neg_abs_logits)),
    name=name)
'''
'''
def binary_crossentropy(labels, logits):
    print("calling binary crossentropy")


    zeros = jnp.zeros_like(logits)
    print("zeros: {}".format(zeros))
    cond = (logits>=zeros)
    print("cond: {}".format(cond))
    relu_logits = jnp.where(cond, logits, zeros)
    print("relu logits: {}".format(relu_logits))
    neg_abs_logits = jnp.where(cond, -logits, logits)
    print("neg abs logits: {}".format(neg_abs_logits))
    return (relu_logits - logits * labels) + (jnp.log(jnp.exp(neg_abs_logits)))
'''
def binary_crossentropy(logits, labels):
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    return -labels * log_p - (1. - labels) * log_not_p

def normalize(x):
    mean = jnp.mean(x)
    var = jnp.std(x)
    
    y = (x - mean) / jnp.sqrt(var + 0.001)
    return y

class Conv2D(object):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_shape
        ):

        self.kernels = [(np.random.rand(*kernel_shape, in_channels, out_channels)-0.5) ]
    
        self.conv_fn = self.make_conv_fn((1,1))
        self.fn = lambda lhs, rhs : normalize(ReLU(self.conv_fn(lhs, rhs)))

    def make_conv_fn(self, stride):
        return lambda lhs, rhs : conv_general_dilated(
            lhs, rhs,
            stride,
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
class ResNetBlock(object):
    def __init__(
        self, 
        channels: int,
        kernel_shape
    ):
        
        self.conv1 = Conv2D(channels, channels, kernel_shape)
        self.conv2 = Conv2D(channels, channels, kernel_shape)
        
        self.fn = self.make_fn()
        self.kernels = self.conv1.kernels + self.conv2.kernels
        
    def make_fn(self):
        return lambda x, k_1, k_2 : x + self.conv2.fn(self.conv2.fn(x, k_1),k_2)

class ConvModel(object):
    def __init__(self):
        
        print("initializing ConvNet...")
        print("using normalization")

        self.modules = [
            Conv2D(3, 16, (3,3)),
            ResNetBlock(16, (3,3)),
            Conv2D(16, 32, (3,3)),
            ResNetBlock(32, (3,3)),
            Conv2D(32, 64, (3,3)),
            ResNetBlock(64, (3,3)),
            Conv2D(64, 128, (3,3)),
            ResNetBlock(128, (3,3)),
            Conv2D(128, 64, (3,3)),
            ResNetBlock(64, (3,3)),
            Conv2D(64, 32, (3,3)),
            ResNetBlock(32, (3,3)),
            Conv2D(32, 1, (3,3)),
        ]
        
        
        self.kernels = list()
        self.model_fn = self.make_model_fn(self.modules)
        
    
    def make_model_fn(self, modules: list):
        if len(modules) == 0:
            return lambda x : x
        
        module = modules[-1]
        module_kernels = module.kernels
        
        num_kernels = len(module_kernels)
        
        upstream_fn = self.make_model_fn(modules[:-1])
        self.kernels += module_kernels
        
        return lambda x, *kernels : module.fn( upstream_fn(x, *kernels[:-num_kernels]), *kernels[-num_kernels:])
        
    
    ''' 
    def make_model_fn(self):
        self.kernels = list()
        
        t_fn = lambda x, *kernels : x
        
        for module in self.modules:
            num_module_kernels = len(module.kernels)
            
            fn = lambda x, *kernels : module.fn( t_fn(x, *kernels[:-num_module_kernels]),  *kernels[-num_module_kernels:])
            t_fn = fn
            fn = None
            
            self.kernels += module.kernels
        return t_fn
            
    '''
    
    def save_kernels(self, dir_path: str, verbose: bool = False):
        print("saving model kernels to {}".format(dir_path))
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        for i, kernel in enumerate(self.kernels):
            with open(os.path.join(dir_path, str(i) + ".npy"), 'wb') as f:
                np.save(f, kernel)
        if verbose:
            print("saved model kernels")
    def load_kernels(self, dir_path: str, verbose: bool = False):
        for i, kernel in enumerate(self.kernels):
            with open(os.path.join(dir_path, str(i) + ".npy"), 'rb') as f:
                self.kernels[i] = np.load(f)
        if verbose:
            print("loaded model kernels")
'''
    @staticmethod
    def make_model_fn(modules: list):

        # I understand that this is a lot more practical
        #fn = lambda x, *k : x
        #for module in modules:
        #    fn = lambda x, *k : module.fn(fn(x, *k[:-1]), k[-1])

        
        fn1 = lambda x, k_1 : modules[0].fn(x, k_1)
        fn2 = lambda x, k_1, k_2 : modules[1].fn(fn1(x, k_1), k_2)
        fn3 = lambda x, k_1, k_2, k_3 : modules[2].fn(fn2(x, k_1, k_2), k_3)
        fn4 = lambda x, k_1, k_2, k_3, k_4 : modules[3].fn(fn3(x, k_1, k_2, k_3), k_4)
        fn5 = lambda x, k_1, k_2, k_3, k_4, k_5 : modules[4].fn(fn4(x, k_1, k_2, k_3, k_4), k_5)
        fn6 = lambda x, k_1, k_2, k_3, k_4, k_5, k_6 : modules[5].fn(fn5(x, k_1, k_2, k_3, k_4, k_5), k_6)
        fn7 = lambda x, k_1, k_2, k_3, k_4, k_5, k_6, k_7 : modules[6].fn(fn6(x, k_1, k_2, k_3, k_4, k_5, k_6), k_7)
        fn8 = lambda x, k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8 : modules[7].fn(fn7(x, k_1, k_2, k_3, k_4, k_5, k_6, k_7), k_8)
        

        return fn8
'''   

