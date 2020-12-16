from python_gdal import *
import os
# from skimage import io
import matplotlib.pyplot as plt
from unet import *
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import numpy as np
# n = 1
file = r'D:/Repos/temp'
tif_file = [r'/tiles/tile_WV3_Pansharpen_11_2016_{}.tif'.format(n+1) for n in range(25)]
mask_file = [r'/masks/mask_{}.tif'.format(n+1) for n in range(25)]


def get_image_data(file):
    bands_data = get_raster_info(file)
    image_data = norma_data(bands_data, norma_methods='min-max')
    return image_data


class CrownDataset(Dataset):
    def __init__(self, tif_file, mask_file, n_random):
        self.tif_file = tif_file
        self.mask_file = mask_file
        self.n_random = n_random

    def __len__(self):
        return len(self.tif_file) * self.n_random

    def __getitem__(self, item):
        i = item // self.n_random
        image_data = get_image_data(os.path.join(file + self.tif_file[i]))
        mask_data = get_raster_info(os.path.join(file + self.mask_file[i]))
        location = random_sample(self.n_random)
        new_item = item % self.n_random
        l = location[new_item]
        patch = torch.from_numpy(image_data[l[0]-99: l[0]+101, l[1]-99: l[1]+101].transpose([2, 0, 1]))
        mask = torch.from_numpy(mask_data[l[0]-99: l[0]+101, l[1]-99: l[1]+101].reshape(200, 200))
        sample = {'patch': patch, 'mask': mask}
        return sample


class Totensor(object):
    def __call__(self, sample):
        patch = sample['patch']
        mask = sample['mask']
        patch = torch.from_numpy(patch.transpose((2, 0, 1)))
        mask = torch.from_numpy(one_hot_encode(2, mask).transpose((2, 0, 1)))
        sample = {'patch': patch,
                  'mask': mask}
        return sample


def random_sample(n):
    x = np.random.randint(99, 899, (n,))
    y = np.random.randint(99, 899, (n,))
    location = [(h, w) for h, w in zip(x, y)]
    return location


crowndataset = CrownDataset(tif_file=tif_file, mask_file=mask_file, n_random=500)
sample = crowndataset[0]
print(sample['patch'].shape)
print(sample['mask'].shape)

#ax = plt.subplot2grid((1, 2), (0, 0))
#ax.imshow(sample['patch'][:, :, 1])
#ax1 = plt.subplot2grid((1, 2), (0, 1))
#ax1.imshow(sample['mask'])
#plt.show()

# print(len(crowndataset))
# print(crowndataset[12499]['patch'].shape)
#for m in range(len(crowndataset)):
#    ax = plt.subplot2grid((1, 2), (0, m))
#    ax.imshow(crowndataset[m]['patch'][:, :, 1])
#    ax1 = plt.subplot2grid((1, 2), (0, m))
#    ax1.imshow(crowndataset[m]['mask'])
#    if m == 0:
#        plt.show()
#        break
# print(crowndataset[12499]['mask'].shape)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = Unet(8, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(unet.parameters(), lr=0.001, momentum=0.9)

# dataset = get_train_samples(n=1)
# print(dataset.keys())

dataload = DataLoader(dataset=crowndataset, batch_size=5)
for b, sample_b in enumerate(dataload):
    patch = sample_b['patch'].to(device=device, dtype=torch.float32)
    mask = sample_b['mask'].to(device=device, dtype=torch.long)
    out = unet(patch)
    loss = criterion(out, mask)
    loss.backward()
    optimizer.step()
    print(b, loss)


# for b in dataload:

#   print(b['label'])
# print(dataset['patch'][0].shape)
# for batch, sample in enumerate(dataload):
#   print(batch, sample['patch'].size())

# for d, l in zip(sample_raster, sample_label):

    #out = unet(d.reshape(1, d.shape[0], d.shape[1], d.shape[2]))
    #loss = criterion(out, l)
    #loss.backward()
    #optimizer.step()
    #print(loss)


# print(sample_raster.shape, sample_label.shape)
# dataset = map(get_train_samples, l)
# axe1 = plt.subplot2grid((1, 2), (0, 0))
# axe1.imshow(d[0, :, :])
# axe2 = plt.subplot2grid((1, 2), (0, 1))
# axe2.imshow(l[0, :, :])
# plt.show()
# io.imshow(dataset['patch'][0][1, :, :])
# plt.show()
