import os
import torch
import random
from munch import Munch
import itertools

from torch.autograd import Variable

from .utils import init_func, WarmupConstantSchedule
from .networks import *

class CycleGAN:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        self.nets, self.optims, self.scheduler, self.buffer = self.build_model(args)

        # Lossess
        self.criterion_GAN = GANLoss(gan_mode=args.gan_loss).to(self.device)
        self.criterion_Cycle = torch.nn.L1Loss()
        self.criterion_Idt = torch.nn.L1Loss()
        self.criterion_Background = torch.nn.L1Loss()

    def build_model(self, args):
        nets = Munch()
        #nets.G_A2B = self.define_network(ResnetGenerator_bang(args.input_nc, args.output_nc, attention=args.attention, n_bottleneck=args.g_bottleneck, n_downsampling=args.g_downsampling, affine=False, last_tanh=args.g_tanh), init_func)
        #nets.G_B2A = self.define_network(ResnetGenerator_bang(args.input_nc, args.output_nc, attention=args.attention, n_bottleneck=args.g_bottleneck, n_downsampling=args.g_downsampling, affine=False, last_tanh=args.g_tanh), init_func)

        nets.G_A2B = self.define_network(ResnetGenerator_bang(args.input_nc, args.output_nc, attention=args.attention, n_downsampling=args.g_downsampling, n_blocks=args.g_bottleneck, output_tanh=args.g_tanh), init_func)
        nets.G_B2A = self.define_network(ResnetGenerator_bang(args.input_nc, args.output_nc, attention=args.attention, n_downsampling=args.g_downsampling, n_blocks=args.g_bottleneck, output_tanh=args.g_tanh), init_func)
        
        nets.D_A = self.define_network(NLayerDiscriminatorSpec(args.input_nc*2), init_func)
        nets.D_B = self.define_network(NLayerDiscriminatorSpec(args.input_nc*2), init_func)       

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

    def save(self, path, epoch=0):
        folder_name = str(epoch) if epoch else 'latest'
        
        save_folder = '{}/{}_model'.format(path, folder_name)
        os.makedirs(save_folder, exist_ok=True)

        for name, model in self.nets.items():
            torch.save(model.state_dict(), '{}/{}.pth'.format(save_folder, name))

    def load(self, path):
        save_folder = '{}/{}'.format(path, self.args.load_name)

        for name, model in self.nets.items():
            model.load_state_dict(torch.load('{}/{}.pth'.format(save_folder, name)))

    def forward_visual(self):
        return [self.real_A, self.fake_B, self.real_B, self.fake_A]

class AttGAN:
    #def __init__(self, input_nc, output_nc, lr=0.0002, log_dir=None, lambda_cycle=10, lambda_idt=5, lambda_background=10, attention=False):        
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        self.nets, self.optims, self.scheduler, self.buffer = self.build_model(args)

        # Lossess
        self.criterion_GAN = GANLoss(gan_mode=args.gan_loss).to(self.device)
        self.criterion_Cycle = torch.nn.L1Loss()
        self.criterion_Idt = torch.nn.L1Loss()
        self.criterion_Background = torch.nn.L1Loss()

    def build_model(self, args):
        nets = Munch()
        #nets.G_A2B = self.define_network(ResnetGenerator_bang(args.input_nc, args.output_nc, attention=args.attention, n_bottleneck=args.g_bottleneck, n_downsampling=args.g_downsampling, affine=False, last_tanh=args.g_tanh), init_func)
        #nets.G_B2A = self.define_network(ResnetGenerator_bang(args.input_nc, args.output_nc, attention=args.attention, n_bottleneck=args.g_bottleneck, n_downsampling=args.g_downsampling, affine=False, last_tanh=args.g_tanh), init_func)

        nets.G_A2B = self.define_network(ResnetGenerator_Att(args.input_nc, args.output_nc, attention=args.attention, n_downsampling=args.g_downsampling, n_blocks=args.g_bottleneck, output_tanh=args.g_tanh, reduction_ratio=args.reduction_ratio), init_func)
        nets.G_B2A = self.define_network(ResnetGenerator_Att(args.input_nc, args.output_nc, attention=args.attention, n_downsampling=args.g_downsampling, n_blocks=args.g_bottleneck, output_tanh=args.g_tanh, reduction_ratio=args.reduction_ratio), init_func)
        
        nets.D_A = self.define_network(NLayerDiscriminatorSpec(args.input_nc*2), init_func)
        nets.D_B = self.define_network(NLayerDiscriminatorSpec(args.input_nc*2), init_func)       

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

        self.fake_B, self.attention_mask_B, self.content_mask_B = self.nets.G_A2B(self.real_A)
        self.fake_A, self.attention_mask_A, self.content_mask_A = self.nets.G_B2A(self.real_B)

        self.cycle_A, _, _ = self.nets.G_B2A(self.fake_B)
        self.cycle_B, _, _ = self.nets.G_A2B(self.fake_A)

        self.idt_A, _, _ = self.nets.G_B2A(self.real_A)
        self.idt_B, _, _ = self.nets.G_A2B(self.real_B)

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

    def save(self, path, epoch=0):
        folder_name = str(epoch) if epoch else 'latest'
        
        save_folder = '{}/{}_model'.format(path, folder_name)
        os.makedirs(save_folder, exist_ok=True)

        for name, model in self.nets.items():
            torch.save(model.state_dict(), '{}/{}.pth'.format(save_folder, name))

    def load(self, path):
        save_folder = '{}/{}'.format(path, self.args.load_name)

        for name, model in self.nets.items():
            model.load_state_dict(torch.load('{}/{}.pth'.format(save_folder, name)))

    def forward_visual(self):
        return [self.real_A, self.fake_B, self.attention_mask_B, self.real_B, self.fake_A, self.attention_mask_A]
        #visual = [self.real_A, self.fake_B, self.attention_mask_B, self.real_B, self.fake_A, self.attention_mask_A]
        #return torch.cat(visual, dim=0)

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

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

        
        