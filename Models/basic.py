import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

def tensor2array(tensors):
    arrays = tensors.detach().to("cpu").numpy()
    return np.transpose(arrays, (0, 2, 3, 1))


def rgb2gray(color_batch):
    #! gray = 0.299*R+0.587*G+0.114*B
    gray_batch = color_batch[:, 0, ...] * 0.299 + color_batch[:, 1, ...] * 0.587 + color_batch[:, 2, ...] * 0.114
    gray_batch = gray_batch.unsqueeze_(1)
    return gray_batch


def getParamsAmount(model):
    params = list(model.parameters())
    count = 0
    for var in params:
        l = 1
        for j in var.size():
            l *= j
        count += l
    return count

def checkAverageGradient(model):
    meanGrad, cnt = 0.0, 0
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            meanGrad += torch.mean(torch.abs(parms.grad))
            cnt += 1
    return meanGrad.item() / cnt


def mark_color_hints(input_grays, target_ABs, gate_maps, kernel_size=3):
    dilated_seeds = dilate_seeds(gate_maps, kernel_size=kernel_size+2)
    dilated_hints = target_ABs*dilated_seeds
    rgb_tensor = lab2rgb(torch.cat((input_grays,dilated_hints), 1))
    #! to highlight the seeds with 1-pixel margin
    binary_map = torch.where(gate_maps>0.01, torch.ones_like(gate_maps), torch.zeros_like(gate_maps))
    tmp_center = dilate_seeds(binary_map, kernel_size=kernel_size)
    margin_mask = dilate_seeds(binary_map, kernel_size=kernel_size+2) - tmp_center
    marked_hints = rgb_tensor*(1.-margin_mask) + dilated_seeds*margin_mask
    return marked_hints

def dilate_seeds(gate_maps, kernel_size=3):
    N,C,H,W = gate_maps.shape
    input_unf = F.unfold(gate_maps, kernel_size, padding=kernel_size//2)
    #! Notice: differentiable? just like max pooling?
    dilated_seeds, _ = torch.max(input_unf, dim=1, keepdim=True)
    output = F.fold(dilated_seeds, output_size=(H,W), kernel_size=1)
    #print('-------', input_unf.shape)
    return output


class Quantize(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.round()
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        inputX = ctx.saved_tensors
        return grad_output


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, convNum):
        super(ConvBlock, self).__init__()
        self.inConv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        layers = []
        for _ in range(convNum - 1):
            layers.append(nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.inConv(x)
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        residual = self.conv(x)
        return x + residual


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, withConvRelu=True):
        super(DownsampleBlock, self).__init__()
        if withConvRelu:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SkipConnection(nn.Module):
    def __init__(self, channels):
        super(SkipConnection, self).__init__()
        self.conv = nn.Conv2d(2 * channels, channels, 1, bias=False)

    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        return self.conv(x)


class Space2Depth(nn.Module):
    def __init__(self, scaleFactor):
        super(Space2Depth, self).__init__()
        self.scale = scaleFactor
        self.unfold = nn.Unfold(kernel_size=scaleFactor, stride=scaleFactor)

    def forward(self, x):
        (N, C, H, W) = x.size()
        y = self.unfold(x)
        y = y.view((N, int(self.scale * self.scale), int(H / self.scale), int(W / self.scale)))
        return y


#! copy from Richard Zhang [SIGGRAPH2017]
# RGB grid points maps to Lab range: L[0,100], a[-86.183,98,233], b[-107.857,94.478]
#------------------------------------------------------------------------------
def rgb2xyz(rgb):  # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        #  [0.212671, 0.715160, 0.072169],
        #  [0.019334, 0.119193, 0.950227]])
    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()
    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)
    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)
    return out

def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])
    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]
    rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
    #！ sometimes reaches a small negative number, which causes NaNs
    rgb = torch.max(rgb,torch.zeros_like(rgb))
    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()
    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)
    return rgb

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()
    xyz_scale = xyz/sc
    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()
    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)
    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)
    return out

def lab2xyz(lab):
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)
    out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()
    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = sc.to(out.device)
    out = out*sc
    return out

def rgb2lab(rgb, l_mean=50, l_norm=50, ab_norm=110):
    #! input rgb: [0,1]
    #! output lab: [-1,1]
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:,[0],:,:]-l_mean) / l_norm
    ab_rs = lab[:,1:,:,:] / ab_norm
    out = torch.cat((l_rs,ab_rs),dim=1)
    return out
    
def lab2rgb(lab_rs, l_mean=50, l_norm=50, ab_norm=110):
    #! input lab: [-1,1]
    #! output rgb: [0,1]
    l_ = lab_rs[:,[0],:,:] * l_norm + l_mean
    ab = lab_rs[:,1:,:,:] * ab_norm
    lab = torch.cat((l_,ab), dim=1)
    out = xyz2rgb(lab2xyz(lab))
    return out