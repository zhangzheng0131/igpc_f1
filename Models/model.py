import torch
import torch.nn as nn
from .network import ResidualHourGlass2, ResNet, ResidualBlock
from .basic import *
from collections import OrderedDict
from RZPack.models import create_model
from RZPack.models import networks
from RZPack.util import util as util_rz
from RZPack.options.train_options import TrainOptions
import os

class EncodeNet(nn.Module):
    def __init__(self, inChannel=3, outChannel=1):
        super(EncodeNet, self).__init__()
        self.net = ResidualHourGlass2(inChannel=inChannel, outChannel=outChannel)

    def forward(self, input_tensor):
        #noise_map = torch.randn_like(torch.unsqueeze(input_tensor[:,1,:,:], 1)) * 0.3
        #output_tensor = self.net(torch.cat((input_tensor, noise_map), dim=1))
        output_tensor = self.net(input_tensor)
        output_tensor = torch.tanh(output_tensor)
        return output_tensor

class EncodeNetNoise(nn.Module):
    def __init__(self, inChannel=4, outChannel=1):
        super(EncodeNetNoise, self).__init__()
        self.net = ResidualHourGlass2(inChannel=inChannel, outChannel=outChannel)

    def forward(self, input_tensor):
        noise_map = torch.randn_like(torch.unsqueeze(input_tensor[:,1,:,:], 1)) * 0.3
        output_tensor = self.net(torch.cat((input_tensor, noise_map), dim=1))
        output_tensor = torch.tanh(output_tensor)
        return output_tensor

class EncodeNetANoise(nn.Module):
    def __init__(self, inChannel=3, outChannel=1):
        super(EncodeNetANoise, self).__init__()
        self.net = ResidualHourGlass2(inChannel=inChannel, outChannel=outChannel)

    def forward(self, input_tensor):
        noise_map = torch.randn_like(torch.unsqueeze(input_tensor[:,1,:,:], 1)) * 0.3
        output_tensor = self.net(input_tensor+noise_map)
        output_tensor = torch.tanh(output_tensor)
        return output_tensor


class DecodeNet(nn.Module):
    def __init__(self, inChannel=1, outChannel=3):
        super(DecodeNet, self).__init__()
        self.net = ResNet(inChannel=inChannel, outChannel=outChannel)

    def forward(self, input_tensor):
        output_tensor = self.net(input_tensor)
        output_tensor = torch.tanh(output_tensor)
        return output_tensor


class DecodeNetPC(nn.Module):
    def __init__(self, inChannel=1, outChannel=3):
        super(DecodeNetPC, self).__init__()
        ## colorization network
        norm_layer = networks.get_norm_layer(norm_type='batch')
        self.colorization_net = networks.SIGGRAPHGenerator(4, 2, norm_layer=norm_layer, use_tanh=True, classification=False)
        self.load_colorizationweights()
        ## new modules
        self.fractor = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=3, padding=1)]\
                                    + [ResidualBlock(64) for _ in range(3)]\
                                    + [nn.Conv2d(64, 4, kernel_size=1, padding=0)]))
    
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
    
    def load_colorizationweights(self):
        checkpts = "../checkpoints/siggraph_caffemodel"
        which_epoch = 'latest'
        name = 'G'
        load_filename = '%s_net_%s.pth' % (which_epoch, name)
        load_path = os.path.join(checkpts, load_filename)
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, self.colorization_net, key.split('.'))
        self.colorization_net.load_state_dict(state_dict)

    def setFrozeLayers(self):
        print('=> frozen decoder backbone')
        for name, param in self.colorization_net.named_parameters():
            if 'model1.0' in name:
                continue
            param.requires_grad = False
        # totally freeze BN in backbone since no other layers have BN
        self.colorization_net.eval()
    
    def releaseFrozeLayers(self):
        print('=> release decoder backbone')
        for name, param in self.colorization_net.named_parameters():
            param.requires_grad = True
        # totally freeze BN in backbone since no other layers have BN
        self.colorization_net.train()

    def train(self, with_frozenLayers=False):
        if with_frozenLayers:
            self.colorization_net.eval()
        else:
            self.fractor.train()
            self.colorization_net.train()

    def forward(self, input_tensor):
        mixed_maps = self.fractor(input_tensor)
        gray_input = mixed_maps[:,0:1,:,:]
        hint_input = mixed_maps[:,1:3,:,:]
        mask_input = mixed_maps[:,3:4,:,:]
        ## vlaue [-0.5,0.5]: intput setting of the pretrained model
        #mask_input = torch.zeros_like(input_tensor)
        #hint_input = torch.zeros_like(hint_input)
        _, pred_ABs = self.colorization_net(input_tensor/2, hint_input, mask_input)
        return pred_ABs


class DecodeNetPC2(nn.Module):
    def __init__(self, inChannel=1, outChannel=3):
        super(DecodeNetPC2, self).__init__()
        ## colorization network
        norm_layer = networks.get_norm_layer(norm_type='batch')
        self.colorization_net = networks.SIGGRAPHGenerator(4, 2, norm_layer=norm_layer, use_tanh=True, classification=False)
        self.load_colorizationweights()
        ## new modules
        self.fractor = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=3, padding=1)]\
                                    + [ResidualBlock(64) for _ in range(3)]\
                                    + [nn.Conv2d(64, 5, kernel_size=1, padding=0)]))
        self.graymapper = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=3, padding=1),\
                                           ResidualBlock(64),\
                                           nn.Conv2d(64, 1, kernel_size=1, padding=0)]))
    
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
    
    def load_colorizationweights(self):
        checkpts = "../checkpoints/siggraph_caffemodel"
        which_epoch = 'latest'
        name = 'G'
        load_filename = '%s_net_%s.pth' % (which_epoch, name)
        load_path = os.path.join(checkpts, load_filename)
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, self.colorization_net, key.split('.'))
        self.colorization_net.load_state_dict(state_dict)

    def setFrozeLayers(self):
        print('=> frozen decoder backbone')
        for name, param in self.colorization_net.named_parameters():
            if 'module.model1.0' in name:
                continue
            param.requires_grad = False
        # totally freeze BN in backbone since no other layers have BN
        self.colorization_net.eval()
    
    def releaseFrozeLayers(self):
        print('=> release decoder backbone')
        for name, param in self.colorization_net.named_parameters():
            param.requires_grad = True
        # totally freeze BN in backbone since no other layers have BN
        self.colorization_net.train()

    def train(self, with_frozenLayers=False):
        if with_frozenLayers:
            self.fractor.train()
            self.colorization_net.eval()
        else:
            self.fractor.train()
            self.colorization_net.train()

    def forward(self, input_tensor):
        mixed_maps = self.fractor(input_tensor)
        gray_input = mixed_maps[:,0:1,:,:]
        hint_input = mixed_maps[:,1:3,:,:]
        mask_input = mixed_maps[:,3:4,:,:]
        pred_Ls = self.graymapper(mixed_maps[:,4:5,:,:])
        ## vlaue [-0.5,0.5]: intput setting of the pretrained model
        #mask_input = torch.zeros_like(input_tensor)
        #hint_input = torch.zeros_like(hint_input)
        _, pred_ABs = self.colorization_net(input_tensor/2, hint_input, mask_input)
        return pred_Ls, pred_ABs


## Part B: discriminative hint location ---------------------------------------------------------------------------
class ParseNetZZ(nn.Module):
    def __init__(self, inChannel=1, outChannel=1, opt=None):
        super(ParseNetZZ, self).__init__()
        self.opt = opt
        self.parser = ResidualHourGlass2(inChannel=inChannel, outChannel=outChannel)
        self.sigmoid = nn.Sigmoid()

    def get_L0norm_mask(self, param_tensor, beta=0.75, gamma=-0.1, zeta=1.1):
        #! Notice this: why sometimes we need to create tensor for scalar but gray_input=gray_input/2 is allowed ?
        #! Guess: cpu scalar is allowed to calculate with a variable tensor (requires_grad=true) BUT not allowed for constant tensor
        zeta = 1.01
        one = torch.tensor([1.0]).cuda(param_tensor.get_device())
        map_size = param_tensor.size()
        #loc_map = param_tensor
        hint_masks = torch.sigmoid(100*(param_tensor-0.01))
        loc_map = torch.log(hint_masks**2 + 1.e-10)
        #! get random samples from uniform distribution
        u = torch.Tensor(map_size).uniform_(0,1)
        u = u.cuda(loc_map.get_device())
        temp = beta if beta is None else nn.Parameter(torch.zeros(1).fill_(beta))
        s = self.sigmoid((torch.log(u + one * 1.e-10) - torch.log(one * (1+1.e-10) - u) + loc_map) / (one * 0.75))
        #! stretch s to [gamma, zeta] from [0,1]
        s = s * (zeta - gamma) + gamma
        penalty = self.sigmoid(loc_map - one * 0.75 * math.log(-gamma / zeta)).sum(dim=(1,2,3), keepdim=True)
        if False and (not self.training):
            # it enters when self.model.eval()
            s = self.sigmoid(loc_map) * (zeta - gamma) + gamma
        #! hard-sigmoid clip
        s = torch.min(torch.max(s, torch.zeros_like(s)), torch.ones_like(s))
        return s, penalty

    def forward(self, input_tensor):
        param_maps = self.parser(input_tensor)
        gate_maps, l0_loss = self.get_L0norm_mask(param_maps)
        return l0_loss, param_maps, gate_maps


class ParseNet(nn.Module):
    def __init__(self, inChannel=1, outChannel=1, opt=None):
        super(ParseNet, self).__init__()
        self.opt = opt
        self.parser = ResidualHourGlass2(inChannel=inChannel, outChannel=outChannel)
        self.sigmoid = nn.Sigmoid()

    def get_L0norm_mask(self, param_tensor, beta=0.75, gamma=-0.1, zeta=1.1):
        #! Notice this: why sometimes we need to create tensor for scalar but gray_input=gray_input/2 is allowed ?
        #! Guess: cpu scalar is allowed to calculate with a variable tensor (requires_grad=true) BUT not allowed for constant tensor
        one = torch.tensor([1.0]).cuda(param_tensor.get_device())
        map_size = param_tensor.size()
        #loc_map = param_tensor
        loc_map = torch.log(param_tensor**2+1.e-10)
        #! get random samples from uniform distribution
        u = torch.Tensor(map_size).uniform_(0,1)
        u = u.cuda(loc_map.get_device())
        temp = beta if beta is None else nn.Parameter(torch.zeros(1).fill_(beta))
        s = self.sigmoid((torch.log(u + one * 1.e-10) - torch.log(one * (1+1.e-10) - u) + loc_map) / (one * 0.75))
        #! stretch s to [gamma, zeta] from [0,1]
        s = s * (zeta - gamma) + gamma
        penalty = self.sigmoid(loc_map - one * 0.75 * math.log(-gamma / zeta)).sum(dim=(1,2,3), keepdim=True)
        if (not self.training):
            # it enters when self.model.eval()
            s = self.sigmoid(loc_map) * (zeta - gamma) + gamma
            #s = loc_map
        #! hard-sigmoid clip
        s = torch.min(torch.max(s, torch.zeros_like(s)), torch.ones_like(s))
        return s, penalty

    def forward(self, input_tensor):
        param_maps = self.parser(input_tensor)
        gate_maps, l0_loss = self.get_L0norm_mask(param_maps)
        return l0_loss, param_maps, gate_maps


class ColorizationModel:
    def __init__(self, checkpts=None, distribute_mode=False, gpu_no=0):
        self.opt = TrainOptions().parse()
        self.isTrain = False
        self.opt.load_model = True
        self.opt.display_id = -1  # no visdom display
        self.opt.checkpoints_dir = checkpts
        if distribute_mode:
            print("@@@Warning: the colorization model does not support distribution mode!")
            self.opt.gpu_ids = gpu_no
        else:
            self.opt.gpu_ids = list(range(torch.cuda.device_count()))        
        self.opt.name = 'siggraph_caffemodel'
        self.opt.mask_cent = 0.0  # NOTICE: mask_cent=0.0 when 'siggraph_caffemodel' used
        self.opt.input_nc = 1
        self.opt.output_nc = 2
        self.colorization_model = create_model(self.opt)
        self.colorization_model.setup(self.opt)
        self.colorization_model.eval()
        print('[*] Colorization model loaded!')
    
    def thresCut(self, soft_mask, threshold=0.95):
        zero = torch.tensor([0.0]).cuda(soft_mask.get_device())
        soft_mask = torch.where(soft_mask > threshold, soft_mask, zero)
        return soft_mask

    def seedDilate(self, color_seed):
        # TBD
        return color_seed

    def testRandomHints(self, gray_input, target_AB, point_num=50):
        #! L rescaling [-0.5,0.5]: intput setting of the pretrained model
        gray_input = gray_input / 2.0
        #! ab: (-1.0,1.0)
        data = {}
        data['A'] = gray_input
        data['B'] = target_AB
        self.opt.sample_Ps = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        data = util_rz.add_color_patches_rand_gt(data, self.opt, p=.125, num_points=point_num, use_avg=True)
        color_seed = data['hint_B'].cuda(gray_input.get_device())
        hint_mask = data['mask_B'].cuda(gray_input.get_device())
        pred_ABs = self.colorization_model.colorize_input(gray_input, color_seed, hint_mask)
        return pred_ABs

    def __call__(self, gray_input, color_seed, hint_mask):
        # option: dilate the color seed
        color_seed = self.seedDilate(color_seed)
        #! L rescaling [-0.5,0.5]: intput setting of the pretrained model
        pred_ABs = self.colorization_model.colorize_input(gray_input/2.0, color_seed, hint_mask)
        return pred_ABs

