import numpy as np
import torch
import torch.nn as nn
from . import util as lib_util


def init_modules_xavier(module_list):
    for m in module_list:
        if isinstance(m, nn.Conv2d) or \
                isinstance(m, nn.ConvTranspose2d) or \
                isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                # print('init')
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Sequential):
            init_modules_xavier(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.LeakyReLU):
            pass
        else:
            raise NameError('ModuleInitError: %s' % str(type(m)))


def __limit_xy__(o1, xy_limit_scale):
    o1_xy, o1_wh = o1[:, :2], o1[:, 2:]
    _o1_xy = torch.tanh(o1_xy) * xy_limit_scale
    _o1 = torch.cat([_o1_xy, o1_wh], dim=1)
    return _o1


def __cvt_xywh2ltrb__(_o1):
    _o1[:, 0] = _o1[:, 0] - (_o1[:, 2] / 2)
    _o1[:, 1] = _o1[:, 1] - (_o1[:, 3] / 2)
    _o1[:, 2] = _o1[:, 0] + _o1[:, 2]
    _o1[:, 3] = _o1[:, 1] + _o1[:, 3]
    return _o1


def create_xy_maps(batch_size, fmap_sizes, coord_range):
    xy_maps = list()
    for fmap_size in fmap_sizes:
        xy_map = lib_util.create_coord_map(fmap_size, coord_range)
        xy_maps.append(torch.from_numpy(xy_map).float().requires_grad_(False))
    xy_maps = [torch.cat([xy_map] * batch_size, dim=0) for xy_map in xy_maps]
    return xy_maps


def create_def_coord(batch_size, output_sizes, coord_range):
    def_coords = list()
    num_def_coords = len(output_sizes)
    for lv, output_size in enumerate(output_sizes):
        def_coord = create_box_coord_map(output_size, num_def_coords, coord_range)[:, :, lv]
        def_coords.append(torch.from_numpy(def_coord).float().view(1, 4, -1).requires_grad_(False))
    def_coord = torch.cat([torch.cat(def_coords, dim=2)] * batch_size, dim=0)
    return def_coord


def create_box_coord_map(output_size, output_ch, coord_range):
    box_coord_map = np.zeros((output_ch, 4, output_size[0], output_size[1])).astype(np.float32)
    box_coord_map[:, :2] += lib_util.create_coord_map(output_size, coord_range)

    # gauss_ch: 4 --> ((0, 1, 2, 3), ...)
    ch_map = np.array(list(range(output_ch)))

    # coord_w: 100 --> unit_intv_w: 20 = 100 / (4 + 1)
    unit_intv_w = coord_range[1] / (output_ch + 1.0)
    unit_intv_h = coord_range[0] / (output_ch + 1.0)

    # ((0, 1, 2, 3) + 1) * 20 == (20, 40, 60, 80)
    w_map = (ch_map + 1) * unit_intv_w
    h_map = (ch_map + 1) * unit_intv_h

    # ((20, 40, 60, 80) / 100)^2 == (0.04, 0.16, 0.36, 0.64)
    # (0.04, 0.16, 0.36, 0.64) * 100 == (4, 16, 36, 64)
    # w_map = ((w_map / coord_range[1]) ** 2) * coord_range[1]
    # h_map = ((h_map / coord_range[0]) ** 2) * coord_range[0]
    w_map = (w_map / coord_range[1]) * coord_range[1]
    h_map = (h_map / coord_range[0]) * coord_range[0]

    w_map = w_map.reshape((output_ch, 1, 1))
    h_map = h_map.reshape((output_ch, 1, 1))
    box_coord_map[:, 2] = w_map
    box_coord_map[:, 3] = h_map

    # (1, 4, gauss_ch, gauss_h, gauss_w)
    box_coord_map = np.transpose(box_coord_map, axes=(1, 0, 2, 3))
    box_coord_map = np.expand_dims(box_coord_map, axis=0)
    return box_coord_map


def create_limit_scale(batch_size, output_sizes, coord_range, limit_factor):
    # n_lv_mix_comps = [output_size[0] * output_size[1] for output_size in output_sizes]

    lv_x_limit_scales, lv_y_limit_scales = list(), list()
    # for i, n_lv_mix_comp in enumerate(n_lv_mix_comps):
    for output_size in output_sizes:
        x_limit_scale = (coord_range[1] / output_size[1]) * limit_factor
        y_limit_scale = (coord_range[0] / output_size[0]) * limit_factor

        n_lv_mix_comp = output_size[0] * output_size[1]
        lv_x_limit_scales.append(x_limit_scale * torch.ones((1, 1, n_lv_mix_comp)).float())
        lv_y_limit_scales.append(y_limit_scale * torch.ones((1, 1, n_lv_mix_comp)).float())

    x_limit_scale = torch.cat(lv_x_limit_scales, dim=2)
    y_limit_scale = torch.cat(lv_y_limit_scales, dim=2)
    limit_scale = torch.cat([x_limit_scale, y_limit_scale], dim=1).requires_grad_(False)
    limit_scale = torch.cat([limit_scale] * batch_size, dim=0)
    return limit_scale
