import os
from munch import Munch
from torchvision.utils import save_image, make_grid
import torch

import pandas as pd

def get_paths(args):
    if (args.save_name == None) and (not args.test):
        args.save_name = '{}_2_{}'.format(args.a_data, args.b_data)
        
    if os.path.exists(args.save_name):
        pass

    paths = Munch()
    paths.save_base = '{}/{}'.format('checkpoints', args.save_name)
    paths.epoch_result = '{}/imgs'.format(paths.save_base)
    paths.test_A2B = '{}/result/{}_2_{}'.format(paths.save_base, args.a_data, args.b_data)
    paths.test_B2A = '{}/result/{}_2_{}'.format(paths.save_base, args.b_data, args.a_data)
    paths.trained = '{}/trained'.format(paths.save_base)
    paths.log_dir = '{}/{}'.format(paths.save_base, 'log')

    for k, v in paths.items():
        os.makedirs(v, exist_ok=True)
    
    return paths

def del_extension(filename):
    return filename[0].split('.')[0]

def save_results(imgs, path, epoch):
    imgs = torch.cat(imgs, dim=0).cpu()
    imgs = imgs * 0.5 + 0.5
    imgs = make_grid(imgs, nrow=4)
    save_image(imgs, '{}/{}_Epoch.png'.format(path, epoch))

def print_options(args, paths):
    args_dict = vars(args)
    df_args = pd.DataFrame.from_dict(args_dict, orient='index', names=['key', 'values'])
    print(df_args.head())
    df_args.to_csv('{}/args.csv'.format(paths.save_base), index=True)
