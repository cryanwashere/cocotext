import numpy as np
import nn


y_true = np.array([0., 0., 0., 1., 1., 1., 0.5])
y_pred = np.array([1., -1., 0., 1., -1., 0., 0.])

print("y_true: {}".format(y_true))
print("y_pred: {}".format(y_pred))

print(nn.binary_crossentropy(y_true, y_pred))