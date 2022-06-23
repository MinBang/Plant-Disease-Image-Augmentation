import os
import torch
import random
from munch import Munch
import itertools

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from .utils import init_func, WarmupConstantSchedule
from .networks import *

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class CycleGAN:
    #def __init__(self, input_nc, output_nc, lr=0.0002, log_dir=None, lambda_cycle=10, lambda_idt=5, lambda_background=10, attention=False):        
    def __init__(self, args, paths):
        self.args = args
        self.paths = paths

        self.train_iter = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.nets, self.optims, self.scheduler, self.buffer = self.build_model(args)

        # Lossess
        self.criterion_GAN = GANLoss(gan_mode=args.gan_loss).to(self.device)
        self.criterion_Cycle = torch.nn.L1Loss()
        self.criterion_Idt = torch.nn.L1Loss()
        self.criterion_Background = torch.nn.L1Loss()

        self.logger = None
        if args.tensorboard:
            self.logger = SummaryWriter(log_dir=paths.log_dir)

        if args.load_name:
            assert (args.save_name is not None), 'you must enter the save_name !!!'
            print('{} Model was Loaded'.format(args.load_name))
            self.load()

            load_epoch = args.load_name.split('_')[0]
            args.start_epoch = 1 if (load_epoch == 'latest') else int(load_epoch)
            args.epochs = args.epochs if (args.start_epoch == 1) else (args.epochs - 1)

    def build_model(self, args):
        nets = Munch()
        nets.G_A2B = self.define_network(ResnetGenerator_bang(args.input_nc, args.output_nc, attention=args.attention, n_bottleneck=args.g_bottleneck, n_downsampling=args.g_downsampling, affine=False, last_tanh=args.g_tanh), init_func)
        nets.G_B2A = self.define_network(ResnetGenerator_bang(args.input_nc, args.output_nc, attention=args.attention, n_bottleneck=args.g_bottleneck, n_downsampling=args.g_downsampling, affine=False, last_tanh=args.g_tanh), init_func)
        
        if args.d_type == 'patch_normal':
            print('NLayerDiscriminatorSpec')
            nets.D_A = self.define_network(NLayerDiscriminatorSpec(args.input_nc*2), init_func)
            nets.D_B = self.define_network(NLayerDiscriminatorSpec(args.input_nc*2), init_func)
        elif args.d_type == 'stargan':
            nets.D_A = self.define_network(Discriminator(args.input_nc*2, attention=args.attention, patch_gan=False), init_func)
            nets.D_B = self.define_network(Discriminator(args.input_nc*2, attention=args.attention, patch_gan=False), init_func)                   
        else:
            nets.D_A = self.define_network(Discriminator_bang(args.input_nc*2, attention=args.attention), init_func)
            nets.D_B = self.define_network(Discriminator_bang(args.input_nc*2, attention=args.attention), init_func)                   

        optims = Munch()
        optims.G = torch.optim.Adam(itertools.chain(nets.G_A2B.parameters(), nets.G_B2A.parameters()), lr=args.lr_G, betas=(args.beta1, args.beta2))
        optims.D = torch.optim.Adam(itertools.chain(nets.D_A.parameters(), nets.D_B.parameters()), lr=args.lr_D, betas=(args.beta1, args.beta2))

        scheduler = Munch()    
        #scheduler.G = WarmupConstantSchedule(optims.G, warmup_steps=args.warmup_steps) 
        #scheduler.D = WarmupConstantSchedule(optims.D, warmup_steps=args.warmup_steps) 

        buffer = Munch()
        buffer.A_replay = ReplayBuffer(max_size=args.buffer_size)
        buffer.B_replay = ReplayBuffer(max_size=args.buffer_size)

        return nets, optims, scheduler, buffer

    def set_input(self, batch):
        self.real_A, self.real_B = batch['A'].to(self.device), batch['B'].to(self.device)
        self.foreground_mask_A, self.foreground_mask_B = batch['A_mask'].to(self.device), batch['B_mask'].to(self.device)
        self.background_mask_A, self.background_mask_B = (1-self.foreground_mask_A).to(self.device), (1-self.foreground_mask_B).to(self.device)

        self.fake_B = self.nets.G_A2B(self.real_A)
        self.fake_A = self.nets.G_B2A(self.real_B)

        self.cycle_A = self.nets.G_B2A(self.fake_B)
        self.cycle_B = self.nets.G_A2B(self.fake_A)

        self.idt_A = self.nets.G_B2A(self.real_A)
        self.idt_B = self.nets.G_A2B(self.real_B)

        self.input_A = torch.cat([self.real_A, self.foreground_mask_A], dim=1)
        self.input_B = torch.cat([self.real_B, self.foreground_mask_B], dim=1)

        self.input_A_fake = torch.cat([self.fake_A, self.foreground_mask_B], dim=1)
        self.input_B_fake = torch.cat([self.fake_B, self.foreground_mask_A], dim=1)

        self.back_real_A = self.background_mask_A * self.real_A
        self.back_real_B = self.background_mask_B * self.real_B

        self.back_fake_A = self.background_mask_B * self.fake_A
        self.back_fake_B = self.background_mask_A * self.fake_B

    def define_network(self, network, init):
        network.to(self.device)
        network.apply(init)

        return network

    def optimize_parameters(self):
        self.set_requires_grad([self.nets.D_A, self.nets.D_B], False)
        self.optims.G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()
        self.optims.G.step()       # update G_A and G_B's weights

        self.set_requires_grad([self.nets.D_A, self.nets.D_B], True)
        self.optims.D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()
        self.optims.D.step()  # update D_A and D_B's weights        

        self.train_iter += 1

    def logging(self):
        logs = {}
        logs['D/loss'] = self.loss_D.item()
        logs['D/loss_D_A'] = self.loss_D_A.item()
        logs['D/loss_D_B'] = self.loss_D_B.item()   

        logs['G/loss'] = self.loss_G.item()
        logs['G/loss_gan'] = self.loss_gan.item()
        #logs['G/loss_G_A'] = self.loss_G_A.item()
        #logs['G/loss_G_B'] = self.loss_G_B.item()
        logs['G/loss_cycle'] = self.loss_cycle.item()
        logs['G/loss_idt'] = self.loss_idt.item()
        logs['G/loss_back'] = self.loss_background.item()
        
        g_lr, d_lr = self.get_lr()
        logs['LR/lr_G'] = g_lr
        logs['LR/lr_D'] = d_lr

        if self.logger:
            for tag, value in logs.items():
                self.logger.add_scalar(tag, value, self.train_iter)
        return logs

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterion_GAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()

        return loss_D

    def backward_D(self):
        fake_A = self.buffer.A_replay.push_and_pop(self.input_A_fake)
        self.loss_D_A = self.backward_D_basic(self.nets.D_A, self.input_A, fake_A)

        fake_B = self.buffer.B_replay.push_and_pop(self.input_B_fake)
        self.loss_D_B = self.backward_D_basic(self.nets.D_B, self.input_B, fake_B)
        self.loss_D = self.loss_D_A + self.loss_D_B / 2

    def backward_G(self):
        self.loss_G_A = self.criterion_GAN(self.nets.D_A(self.input_A_fake), True) ## Pair netG_B2A
        self.loss_G_B = self.criterion_GAN(self.nets.D_B(self.input_B_fake), True) ## Pair netG_B2A
        self.loss_gan = self.loss_G_A + self.loss_G_B

        self.loss_cycle_A = self.criterion_Cycle(self.cycle_A, self.real_A) * self.args.lambda_cycle
        self.loss_cycle_B = self.criterion_Cycle(self.cycle_B, self.real_B) * self.args.lambda_cycle
        self.loss_cycle = self.loss_cycle_A + self.loss_cycle_B

        self.loss_idt_A = self.criterion_Idt(self.idt_A, self.real_A) * self.args.lambda_idt
        self.loss_idt_B = self.criterion_Idt(self.idt_B, self.real_B) * self.args.lambda_idt
        self.loss_idt = self.loss_idt_A + self.loss_idt_B

        self.loss_background_A = self.criterion_Background(self.back_fake_B, self.back_real_A) * self.args.lambda_background
        self.loss_background_B = self.criterion_Background(self.back_fake_A, self.back_real_B) * self.args.lambda_background
        self.loss_background = self.loss_background_A + self.loss_background_B

        self.loss_G = self.loss_gan + self.loss_cycle + self.loss_idt + self.loss_background
        self.loss_G.backward()

    def update_lr(self):
        for k, v in self.scheduler.items():
            v.step()

    def get_lr(self):
        g_lr = self.optims.G.param_groups[0]['lr']
        d_lr = self.optims.D.param_groups[0]['lr']

        return g_lr, d_lr

    def train(self):
        for name, model in self.nets.items():
            model.train()

    def eval(self):
        for name, model in self.nets.items():
            model.eval()

    def save(self, epoch=0):
        folder_name = str(epoch) if epoch else 'latest'
        
        save_folder = '{}/{}_model'.format(self.paths.trained, folder_name)
        os.makedirs(save_folder, exist_ok=True)

        for name, model in self.nets.items():
            torch.save(model.state_dict(), '{}/{}.pth'.format(save_folder, name))

    def load(self):
        save_folder = '{}/{}'.format(self.paths.trained, self.args.load_name)
        #self.train_iter = 0

        for name, model in self.nets.items():
            model.load_state_dict(torch.load('{}/{}.pth'.format(save_folder, name)))

        
        