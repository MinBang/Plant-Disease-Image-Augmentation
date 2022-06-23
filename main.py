import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import ImageDataset

from models import CycleGAN
from utils import make_paths, del_extension, save_results, print_options, get_config

from torchinfo import summary

def train_epoch(model, dataloader, n_sample, epoch):
    model.train()
    batches = []
    steps_per_epoch = len(dataloader)

    pbar = tqdm(dataloader, desc='Iter Loop', leave=False)
    for i, batch in enumerate(pbar):
        model.set_input(batch)
        model.optimize_parameters()

        logs = model.logging()

        if len(batches) < n_sample:
            batches.append(batch)

        pbar.set_description("Loss_D: {:.4f} || Loss_G: {:.4f} || Lr: ({:.6f})  ".format(logs['D/loss'], logs['G/loss_gan'], logs['LR/lr_D']))
    model.update_lr()
    return batches

def test_epoch(model, dataloader, save_paths):
    model.eval()

    pbar = tqdm(dataloader, desc='Iter Loop', leave=False)
    for i, batch in enumerate(pbar):
        model.set_input(batch)
        fake_A, fake_B = model.fake_A.cpu(), model.fake_B.cpu()
        fake_A = fake_A * 0.5 + 0.5
        fake_B = fake_B * 0.5 + 0.5

        filename_A = del_extension(batch['A_filename'])
        filename_B = del_extension(batch['B_filename'])
        
        save_image(fake_B, '{}/{}.png'.format(save_paths.test_A2B, filename_A))
        save_image(fake_A, '{}/{}.png'.format(save_paths.test_B2A, filename_B))

def test_sample(model, batches, save_paths, epoch):
    model.eval()
    imgs = []

    for i, batch in enumerate(batches):
        model.set_input(batch)
        
        img = torch.cat([model.real_A, model.fake_B, model.real_B, model.fake_A], dim=0)
        imgs.append(img)        
    save_results(imgs, save_paths.epoch_result, epoch)

##########################

def dummy_args(args):
    args.capacity = 1000
    args.data_root = '../../stargan/dataset/Potato_mask'
    args.n_sample = 5
    args.img_size = 224
    args.g_downsampling = 3
    args.g_bottleneck = 6
    args.in_memory = True

    args.save_name = 'Potato_mydis2'
    args.d_type = 'my'
    args.g_tanh = False

    #args.load_name = '3_model'

def print_model(model, args):
    g_input = (1, 3, args.img_size, args.img_size)
    d_input = (1, 6, args.img_size, args.img_size)
    
    summary(model.nets.G_A2B, g_input)
    summary(model.nets.D_A, d_input)

if __name__ == '__main__':
    args = get_config()
    dummy_args(args) ## To be deleted

    save_paths = make_paths(args)
    print_options(args, save_paths)
    
    dataset = ImageDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = CycleGAN(args, save_paths)
    #print_model(model, args)

    if not args.test:
        print('\n ==Train Step== ')
        for epoch in tqdm(range(args.start_epoch, args.epochs+1), desc='Epoch Loop'):
            batches = train_epoch(model, dataloader, args.n_sample, epoch)

            if args.save_name:
                if ((epoch % args.interval_test) == 0) or (epoch==1):
                    test_sample(model, batches, save_paths, epoch)

                if (epoch % args.interval_save == 0) and (args.start_save < epoch):
                    model.save(epoch)
                model.save()
    else:
        print('\n ==Test Step== ')
        assert (args.save_name is not None) and (args.load_name is not None), 'you must enter the save_name and load_name'
        test_epoch(model, dataloader, save_paths)