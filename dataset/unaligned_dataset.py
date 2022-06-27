import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

import albumentations

class ImageDataset(Dataset):
    #def __init__(self, args.data_root, unaligned=False, A='trainA', B='trainB', args.capacity=None, swap=False, isTest=False, args.img_size=256, args.in_memory=False):
    def __init__(self, args):
        self.transform = albumentations.Compose([
            albumentations.Resize(width=int(args.img_size*1.12), height=int(args.img_size*1.12)),
            albumentations.RandomCrop(width=args.img_size, height=args.img_size, p=0.6),
            albumentations.HorizontalFlip(p=0.5),
        ])

        self.torch_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

        self.args = args

        self.files_A = sorted(glob.glob(args.path_A + '/*.*'))
        self.files_B = sorted(glob.glob(args.path_B + '/*.*'))
        
        self.files_A_mask = sorted(glob.glob(args.path_A + '_mask/*.*'))
        self.files_B_mask = sorted(glob.glob(args.path_B + '_mask/*.*'))        

        if self.args.capacity:
            args.capacity_A = min(len(self.files_A), args.capacity)
            args.capacity_B = min(len(self.files_B), args.capacity)

            self.files_A = self.files_A[:args.capacity_A]
            self.files_B = self.files_B[:args.capacity_B]

            self.files_A_mask = self.files_A_mask[:args.capacity_A]
            self.files_B_mask = self.files_B_mask[:args.capacity_B]

        if self.args.in_memory:
            self.files_A_img = [Image.open(f).convert("RGB") for f in self.files_A]
            self.files_B_img = [Image.open(f).convert("RGB") for f in self.files_B]
            
            self.files_A_mask_img = [Image.open(f).convert("RGB") for f in self.files_A_mask]
            self.files_B_mask_img = [Image.open(f).convert("RGB") for f in self.files_B_mask]

    def __getitem__(self, index):
        idx_a = index % len(self.files_A)
        idx_b = index % len(self.files_B)
        if self.args.unaligned:
            idx_b = random.randint(0, len(self.files_B) - 1)

        filename_a, filename_b = self.files_A[idx_a].split('\\')[-1], self.files_B[idx_b].split('\\')[-1]
        filename_a_mask, filename_b_mask = self.files_A_mask[idx_a].split('\\')[-1], self.files_B_mask[idx_b].split('\\')[-1]
        
        if self.args.in_memory:
            item_A, mask_A = self.get_img(self.files_A_img, self.files_A_mask_img, idx_a)
            item_B, mask_B = self.get_img(self.files_B_img, self.files_B_mask_img, idx_b)
        else:
            item_A, mask_A = self.get_img(self.files_A, self.files_A_mask, idx_a)
            item_B, mask_B = self.get_img(self.files_B, self.files_B_mask, idx_b)

        return {'A': item_A, 'B': item_B, 'A_mask':mask_A, 'B_mask':mask_B, 'A_filename':filename_a, 'B_filename':filename_b}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def get_img(self, files, files_mask, idx):
        if self.args.in_memory:
            image = files[idx].copy()
            mask = files_mask[idx].copy()
        else:
            image = Image.open(files[idx]).convert("RGB")
            mask = Image.open(files_mask[idx]).convert("RGB")
        
        if not self.args.test:
            augmentations = self.transform(image=np.array(image), mask=np.array(mask))
            image = Image.fromarray(augmentations["image"])
            mask = Image.fromarray(augmentations["mask"])

        augmentation_img = self.torch_transform(image)
        augmentation_mask = self.torch_transform(mask)

        augmentation_mask = (augmentation_mask >= 0.5).float()

        return augmentation_img, augmentation_mask
