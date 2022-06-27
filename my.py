from sklearn.manifold import TSNE
#from torch.utils.data import ImageDataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

path = '../Potato_test'

trsf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])


dataset = ImageFolder(path, transform=trsf)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

print(len(dataloader))
imgs = [ img.numpy() for img, label in dataloader]
imgs = np.stack(imgs).reshape(len(imgs), -1)
#labels = dataset.targets
print(dataset.class_to_idx)

tsne = TSNE()
imgs_transfomed = tsne.fit_transform(imgs)
labels = dataset.targets

print(imgs_transfomed.shape)

#labels = ['Origin', 'Others', 'PDGAN']
labels = list(dataset.class_to_idx.keys())
size = 152
plt.figure(figsize=(16, 10))
for i in range(3):
    start = i * size
    end = start + size
    plt.scatter(imgs_transfomed[start:end, 0], imgs_transfomed[start:end, 1], alpha=0.6, cmap='Spectral', label=labels[i])
plt.legend()

plt.show()
plt.savefig('test.png')

#imgs = [ transform(batch[0]) for batch in dataloader ]
#print(imgs[0].shape, imgs[-1].shape)

#
#print(imgs.shape)

'''
imgs = np.stack(imgs).reshape(len(imgs), -1)


'''