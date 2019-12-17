import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from lib.network import MMODNetwork
from lib.post_proc import MMODPostProc
from lib.framework import MMODFramework
from lib.loss_func import MMODLossFunction
from lib.dataset import get_dataset_dict
from lib.tester import get_tester_dict


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bash_file', type=str)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--load_dir', type=str, default=None)

    parser.add_argument('--global_args', type=str)
    parser.add_argument('--network_args', type=str)
    parser.add_argument('--post_proc_args', type=str)
    parser.add_argument('--loss_func_args', type=str)
    parser.add_argument('--optimizer_args', type=str)

    parser.add_argument('--train_data_loader_info', type=str)
    parser.add_argument('--test_data_loader_info', type=str)
    parser.add_argument('--tester_info_list', type=str)
    parser.add_argument('--training_args', type=str)

    parser.add_argument('--snapshot_iters', type=str, default='')
    parser.add_argument('--test_iters', type=str, default='')

    args = parser.parse_args()
    args.global_args = cvt_str2python_data(args.global_args)
    args.network_args = cvt_str2python_data(args.network_args)
    args.post_proc_args = cvt_str2python_data(args.post_proc_args)
    args.loss_func_args = cvt_str2python_data(args.loss_func_args)
    args.optimizer_args = cvt_str2python_data(args.optimizer_args)

    args.train_data_loader_info = cvt_str2python_data(args.train_data_loader_info)
    args.test_data_loader_info = cvt_str2python_data(args.test_data_loader_info)
    args.tester_info_list = cvt_str2python_data(args.tester_info_list)

    args.training_args = cvt_str2python_data(args.training_args)
    args.snapshot_iters = cvt_str2python_data(args.snapshot_iters)
    args.test_iters = cvt_str2python_data(args.test_iters)

    assert args.global_args['main_device'] in args.global_args['devices']

    loss_func = create_loss_func(args.global_args, args.loss_func_args)
    network = create_network(args.global_args, args.network_args, loss_func)
    network.build()
    post_proc = create_post_proc(args.global_args, args.post_proc_args)
    framework = create_framework(args.global_args, network, post_proc)
    optimizer = create_optimizer(args.optimizer_args, network)

    data_loader_dict = {
        'train': create_data_loader(args.global_args, args.train_data_loader_info),
        'test': create_data_loader(args.global_args, args.test_data_loader_info)}

    tester_dict = dict()
    for tester_info in args.tester_info_list:
        tester_dict[tester_info['tester']] = create_tester(args.global_args, tester_info)

    if (args.load_dir is not None) and os.path.exists(args.load_dir):
        network_path = os.path.join(args.load_dir, 'network.pth')
        network.load(network_path)

        optimizer_path = os.path.join(args.load_dir, 'optimizer.pth')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
            print('[OPTIMIZER] load: %s' % optimizer_path)
    return args, framework, optimizer, data_loader_dict, tester_dict


def cvt_str2python_data(arg_str):
    if isinstance(arg_str, str):
        python_data = yaml.full_load(arg_str)
    else:
        python_data = arg_str

    if isinstance(python_data, dict):
        for key, value in python_data.items():
            if value == 'None':
                python_data[key] = None
            elif isinstance(value, dict) or isinstance(value, list):
                python_data[key] = cvt_str2python_data(value)

    elif isinstance(python_data, list):
        for i, value in enumerate(python_data):
            if value == 'None':
                python_data[i] = None
            elif isinstance(value, dict) or isinstance(value, list):
                python_data[i] = cvt_str2python_data(value)
    return python_data


def create_loss_func(global_args, loss_func_args):
    return MMODLossFunction(global_args, loss_func_args)


def create_network(global_args, network_args, loss_func):
    return MMODNetwork(global_args, network_args, loss_func)


def create_post_proc(global_args, post_proc_args):
    return MMODPostProc(global_args, post_proc_args)


def create_framework(global_args, network, post_proc):
    return MMODFramework(global_args, network, post_proc)


def create_optimizer(optimizer_args, network):
    optimizer_args.update({'params': network.parameters()})
    return torch.optim.SGD(**optimizer_args)


def create_data_loader(global_args, data_loader_info):
    dataset_key = data_loader_info['dataset']
    dataset_args = data_loader_info['dataset_args']

    batch_size = data_loader_info['batch_size'] \
        if 'batch_size' in data_loader_info.keys() else global_args['batch_size']
    shuffle = data_loader_info['shuffle']
    num_workers = data_loader_info['num_workers']

    dataset = get_dataset_dict()[dataset_key](global_args, dataset_args)
    data_loader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader


def create_tester(global_args, tester_info):
    tester_class = get_tester_dict()[tester_info['tester']]
    return tester_class(global_args, tester_info['tester_args'])
