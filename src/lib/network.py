import abc
import math
import torch
import torch.nn as nn
from torch.nn import functional as func
from .backbone import get_backbone_dict
from . import network_util as net_util
from lib import util as lib_util


class NetworkABC(abc.ABC, nn.Module):
    def __init__(self, global_args, network_args, loss_func):
        super(NetworkABC, self).__init__()
        self.global_args = global_args
        self.network_args = network_args
        self.loss_func = loss_func
        self.net = nn.ModuleDict()

    @ abc.abstractmethod
    def build(self):
        pass

    def save(self, save_path):
        if self.net is not None:
            torch.save(self.net.state_dict(), save_path)
            print('[NETWORK] save: %s' % save_path)

    def load(self, load_path):
        if self.net is not None:
            self.net.load_state_dict(torch.load(load_path, map_location='cpu'))
            print('[NETWORK] load: %s' % load_path)


class MMODNetwork(NetworkABC):
    def __init__(self, global_args, network_args, loss_func):
        super(MMODNetwork, self).__init__(global_args, network_args, loss_func)

        self.coord_range = (global_args['coord_h'], global_args['coord_w'])
        self.img_size = (global_args['img_h'], global_args['img_w'])
        self.main_device = global_args['main_device']
        self.batch_size = global_args['batch_size']
        self.n_classes = global_args['n_classes']

        self.limit_factor = network_args['limit_factor']
        self.std_factor = network_args['std_factor']
        self.fmap_ch = network_args['fmap_ch']

        self.xy_maps = None
        self.def_coord = None
        self.limit_scale = None

    def build(self):
        out_ch = 1 + 4 + 4 + self.n_classes
        backbone = get_backbone_dict()[self.network_args['backbone']](self.network_args)
        backbone.build()

        self.net['backbone'] = backbone
        self.net['detector'] = nn.Sequential(
            nn.Conv2d(self.fmap_ch + 2, self.fmap_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.fmap_ch, self.fmap_ch, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.fmap_ch, self.fmap_ch, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.fmap_ch, out_ch, 1, 1, 0, bias=True))

        output_sizes = list()
        for fmap2img_ratio in backbone.get_fmap2img_ratios():
            output_h = math.ceil(self.img_size[0] * fmap2img_ratio)
            output_w = math.ceil(self.img_size[1] * fmap2img_ratio)
            output_sizes.append((output_h, output_w))

        self.xy_maps = net_util.create_xy_maps(self.batch_size, output_sizes, self.coord_range)
        self.def_coord = net_util.create_def_coord(self.batch_size, output_sizes, self.coord_range)
        self.limit_scale = net_util.create_limit_scale(
            self.batch_size, output_sizes, self.coord_range, self.limit_factor)

        self.net.cuda(self.main_device)
        self.xy_maps = [xy_map.cuda(self.main_device) for xy_map in self.xy_maps]
        self.def_coord = self.def_coord.cuda(self.main_device)
        self.limit_scale = self.limit_scale.cuda(self.main_device)

    def __sycn_batch_and_device__(self, batch_size, device_idx):
        if self.main_device != device_idx:
            xy_maps = [xy_map[:batch_size].cuda(device_idx) for xy_map in self.xy_maps]
            def_coord = self.def_coord[:batch_size].cuda(device_idx)
            limit_scale = self.limit_scale[:batch_size].cuda(device_idx)
        else:
            xy_maps = self.xy_maps[:batch_size]
            def_coord = self.def_coord[:batch_size]
            limit_scale = self.limit_scale[:batch_size]
        return xy_maps, def_coord, limit_scale

    def forward(self, image, boxes=None, labels=None, n_boxes=None, loss=False):
        batch_size = image.shape[0]
        xy_maps, def_coord, xy_limit_scale = \
            self.__sycn_batch_and_device__(batch_size, image.device.index)

        fmaps = self.net['backbone'].forward(image)
        o1, o2, o3, o4 = list(), list(), list(), list()
        for i, fmap in enumerate(fmaps):
            fmap = torch.cat([fmap, xy_maps[i]], dim=1)
            out_params = self.net['detector'].forward(fmap)

            o1.append(out_params[:, 0:4].view((batch_size, 4, -1)))
            o2.append(out_params[:, 4:8].view((batch_size, 4, -1)))
            o3.append(out_params[:, 9:].view(batch_size, self.n_classes, -1))
            o4.append(out_params[:, 8:9].view((batch_size, 1, -1)))

        o1, o2 = torch.cat(o1, dim=2), torch.cat(o2, dim=2)
        o3, o4 = torch.cat(o3, dim=2), torch.cat(o4, dim=2)

        _o1 = net_util.__limit_xy__(o1, xy_limit_scale) + def_coord
        _o1_wh = _o1[:, 2:].clone().detach()
        _o1_whwh = torch.clamp_min(torch.cat([_o1_wh, _o1_wh], dim=1), lib_util.epsilon)
        mu = net_util.__cvt_xywh2ltrb__(_o1)

        sig = func.softplus(o2) + (_o1_whwh * self.std_factor)
        prob = func.softmax(o3, dim=1)
        pi = func.softmax(o4, dim=2)

        if (loss is False) or (boxes is None) or (labels is None) or (n_boxes is None):
            return mu, sig, prob, pi
        else:
            mog_nll_loss, sample_comb_nll_loss = \
                self.loss_func.forward(mu, sig, prob, pi, boxes, labels, n_boxes)
            return mog_nll_loss, sample_comb_nll_loss
