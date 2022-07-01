import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import ImageDataset

from models import CycleGAN
from utils import *

from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import glob
from pprint import pprint

class PDGAN_Solver:
    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_params = ['img_size', 'input_nc', 'output_nc', 'attention', 'lr_G', 'lr_D', 'beta1', 'beta2', 'buffer_size', 'gan_loss', 'g_downsampling', 
                    'g_bottleneck', 'g_tanh', 'd_type', 'reduction_ratio', 'lambda_cycle', 'lambda_idt', 'lambda_background', 'test']

        self.test_mode = args.test
        self.args = self.rectify_args(args)
        self.paths = get_paths(self.args)
        self.train_iter = 0

        self.write_options()
        print(' Save Folder: {}'.format(self.args.save_name))
        print(' Mapping: [ {} <---> {} ]\n'.format(self.args.a_data, self.args.b_data))

        self.dataset = ImageDataset(self.args)
        self.dataloader = DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)
        #print(len(self.dataset), len(self.dataloader))

        self.model = CycleGAN(self.args, self.device)
        self.model_build()

        self.logger = None
        if args.tensorboard:
            self.logger = SummaryWriter(log_dir=self.paths.log_dir)

        self.batches = []
        self.imgs = []

    def run(self):
        pprint(self.args.__dict__, indent=2)

        if self.test_mode:
            print(args.save_name, args.load_name)
            print(' ==Test Step== ')
            self.test()
        else:
            print(' ==Train Step== ')
            print()
            self.train()

    def __train_epoch(self):
        self.model.train()
        self.batches = []

        pbar = tqdm(self.dataloader, desc='Iter Loop', leave=False)
        for i, batch in enumerate(pbar):
            self.model.set_input(batch)
            self.model.optimize_parameters()

            logs = self.model.logging()
            self.log_wrtie(logs)

            if len(self.batches) < self.args.n_sample:
                self.batches.append(batch)

            pbar.set_description("Loss_D: {:.4f} || Loss_G: {:.4f} || Lr: ({:.6f})  ".format(logs['D/loss'], logs['G/loss_gan'], logs['LR/lr_D']))
        self.model.update_lr()

    def __test_sample(self, epoch):
        self.model.eval()
        self.imgs = []
        nrow = len(self.model.forward_visual())

        for i, batch in enumerate(self.batches):
            with torch.set_grad_enabled(False):
                self.model.set_input(batch)
            self.imgs.append(torch.cat(self.model.forward_visual(), dim=0))
        save_results(self.imgs, self.paths.epoch_result, epoch, nrow)

    def train(self):
        for epoch in tqdm(range(self.args.start_epoch, self.args.epochs+1), desc='Epoch Loop'):
            self.__train_epoch()

            if args.save_name:
                if ((epoch % args.interval_test) == 0) or (epoch==1):
                    self.__test_sample(epoch)

                if args.save:
                    if (epoch % args.interval_save == 0) and (args.start_save <= epoch):
                        self.model.save(self.paths.trained, epoch)
                    self.model.save(self.paths.trained)

    def test(self):
        self.model.eval()

        pbar = tqdm(self.dataloader, desc='Iter Loop', leave=False)
        for i, batch in enumerate(pbar):
            self.model.set_input(batch)

            fake_A, fake_B = self.model.fake_A.cpu(), self.model.fake_B.cpu()
            fake_A = fake_A * 0.5 + 0.5
            fake_B = fake_B * 0.5 + 0.5

            filename_A = del_extension(batch['A_filename'])
            filename_B = del_extension(batch['B_filename'])
            
            save_image(fake_B, '{}/{}.png'.format(self.paths.test_A2B, filename_A))
            save_image(fake_A, '{}/{}.png'.format(self.paths.test_B2A, filename_B))

    def log_wrtie(self, logs):
        self.train_iter += 1
        if self.logger:
            for tag, value in logs.items():
                self.logger.add_scalar(tag, value, self.train_iter)
    
    def rectify_args(self, args):
        if args.data_swap:
            args.a_data, args.b_data = args.b_data, args.a_data
        print(args.data_root)

        paths = glob.glob(args.data_root + '/*')
        paths = list(filter(lambda x: 'mask' not in x, paths))

        assert len(paths) != 0, 'Please check the data_root path.'
        print(paths)

        path_A = list(filter(lambda x: args.a_data.upper() in x.upper(), paths))
        path_B = list(filter(lambda x: args.b_data.upper() in x.upper(), paths))
        assert (len(path_A) == 1) and (len(path_B) == 1), 'Please check for possible duplicate folder names in data_root'

        args.path_A, args.path_B = path_A[0], path_B[0]

        args.a_data = args.path_A.split('\\')[-1]
        args.b_data = args.path_B.split('\\')[-1]

        if args.save_name == None:
            args.save_name = '{}_2_{}'.format(args.a_data, args.b_data)
        
        return args

    def write_options(self):
        if args.load_name:
            load_params = json.load(open(self.paths.args, 'r'))
            for k in self.model_params:
                print('{}: {}'.format(k, load_params[k]))
                args.__setattr__(k, load_params[k])

        if not args.test:
            with open(self.paths.args, 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def model_build(self):
        if self.args.load_name:
            print('{} Model was Loaded'.format(self.args.load_name))
            self.model.load(self.paths.trained)

            load_epoch = self.args.load_name.split('_')[0]
            self.args.start_epoch = 1 if (load_epoch == 'latest') else int(load_epoch)
            self.args.epochs = self.args.epochs if (self.args.start_epoch == 1) else (self.args.epochs - 1)

def dummy_args(args):
    args.data_root = 'datasets/Potato_leaf'
    args.a_data = 'early'
    args.b_data = 'healthy'
    
    #args.save_name = 'Att_GAN4'
    #args.load_name = 'latest_model'
    #args.save = False
    #args.test = False

    #args.save_name = 'test'
    #args.save_name = 'Potato___Early_blight_2_Potato___healthy'
    #args.load_name = 'latest_model'

    #args.in_memory = True
    
    #args.n_downsampling = 4
    #args.n_sample = 5
    #args.reduction_ratio = 16
    #args.beta1 = 0.5

    #args.save_name = 'test2'
    #args.d_type = 'my'
    #args.g_tanh = False

    #args.load_name = '3_model'

if __name__ == '__main__':
    args = get_config()
    dummy_args(args)

    solver = PDGAN_Solver(args)
    solver.run()
