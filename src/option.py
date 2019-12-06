import yaml
import argparse
from .lib.network import MMODNetwork
from .lib.post_proc import MMODPostProc
from .lib.loss_func import MMODLossFunction


def parse_mmod_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bash_file', type=str)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--load_dir', type=str, default=None)

    parser.add_argument('--global_args', type=str)
    parser.add_argument('--network_args', type=str)
    parser.add_argument('--postproc_args', type=str)
    parser.add_argument('--lossfunc_args', type=str)
    parser.add_argument('--optim_args', type=str)

    parser.add_argument('--train_dataset_info', type=str)
    parser.add_argument('--test_dataset_info', type=str)
    parser.add_argument('--tester_info_list', type=str)
    parser.add_argument('--train_args', type=str)

    parser.add_argument('--snapshot_iters', type=str, default='')
    parser.add_argument('--test_iters', type=str, default='')

    args = parser.parse_args()
    args.global_args = cvt_str2python_data(args.global_args)
    args.network_args = cvt_str2python_data(args.network_args)
    args.post_proc_args = cvt_str2python_data(args.post_proc_args)
    args.loss_func_args = cvt_str2python_data(args.loss_func_args)
    args.optimizer_args = cvt_str2python_data(args.optimizer_args)

    args.train_data_set_info = cvt_str2python_data(args.train_data_set_info)
    args.test_data_set_info = cvt_str2python_data(args.test_data_set_info)
    args.tester_info_list = cvt_str2python_data(args.tester_info_list)
    args.train_args = cvt_str2python_data(args.train_args)

    args.lr_decay_schd_dict = cvt_str2python_data(args.lr_decay_schd_dict)
    args.lw_schd_dict = cvt_str2python_data(args.lw_schd_dict)
    args.snapshot_iters = cvt_str2python_data(args.snapshot_iters)
    args.test_iters = cvt_str2python_data(args.test_iters)
    return args


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


def create_optimizer(global_args, optimizer_args):
    return None


def create_data_loader(global_args, data_set_info):
    return None
