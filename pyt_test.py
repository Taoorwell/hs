import torch
import numpy as np
# import tensorflow as tf

# print(torch.is_tensor(tensor_a))
# print(tf.is_tensor(tensor_a))
tensor_a = torch.randn(50, 50).reshape(1, 1, 50, 50)
print(tensor_a)

down_sample = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1)(tensor_a)
down_max_pool = torch.nn.MaxPool2d(2, stride=2)(down_sample)
print(down_max_pool.shape)

print(down_sample.shape)
up_sample = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=1)(down_sample)

print(up_sample.shape)
