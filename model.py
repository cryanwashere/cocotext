import os

import jax
import numpy as np
import jax.numpy as jnp
from jax.lax import conv_general_dilated

def ReLU(x):
    return jnp.maximum(x,0)


class Conv2D(object):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_shape
        ):

        self.kernel = np.random.rand(*kernel_shape, in_channels, out_channels) * 0.01
    
        self.conv_fn = self.make_conv_fn((1,1))
        self.fn = lambda lhs, rhs : ReLU(self.conv_fn(lhs, rhs))

    def make_conv_fn(self, stride):
        return lambda lhs, rhs : conv_general_dilated(
            lhs, rhs,
            stride,
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC")
        )


        


class ConvModel(object):
    def __init__(self):

        self.modules = [
            Conv2D(3, 16, (3,3)),
            Conv2D(16, 32, (3,3)),
            Conv2D(32, 64, (3,3)),
            Conv2D(64, 128, (3,3)),
            Conv2D(128, 64, (3,3)),
            Conv2D(64, 32, (3,3)),
            Conv2D(32, 16, (3,3)),
            Conv2D(16, 1, (3,3)),
        ]

        self.model_fn = self.make_model_fn(self.modules)
        self.kernels = [module.kernel for module in self.modules]

    def save_kernels(self, dir_path: str):
        print("saving model kernels to {}".format(dir_path))
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        for i, kernel in enumerate(self.kernels):
            with open(os.path.join(dir_path, str(i) + ".npy"), 'wb') as f:
                np.save(f, kernel)
    def load_kernels(self, dir_path: str):
        for i, kernel in enumerate(self.kernels):
            with open(os.path.join(dir_path, str(i) + ".npy"), 'rb') as f:
                self.kernels[i] = np.load(f)
            
        
    
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
            

