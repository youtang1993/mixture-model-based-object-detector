import os
import shutil
import torch
import torch.nn as nn
from .lib import util as lib_util


def cvt_dict2str(value_dict):
    result_str = ''
    for key, value in value_dict.items():
        result_str += ('%s: %.7f, ' % (key, value))
    result_str = result_str[:-2]
    return result_str.rstrip()


def create_result_dir(result_dir, names=None):
    result_dir_dict = dict()
    lib_util.make_dir(result_dir)
    for name in names:
        dir_path = os.path.join(result_dir, name)
        lib_util.make_dir(dir_path)
        result_dir_dict[name] = dir_path
    return result_dir_dict


def copy_file(src_path, dst_dir):
    src_file = src_path.split('/')[-1]
    dst_path = os.path.join(dst_dir, src_file)
    shutil.copyfile(src_path, dst_path)


def copy_dir(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def save_snapshot(network, optimizer, save_dir):
    lib_util.make_dir(save_dir)
    network_path = os.path.join(save_dir, 'network.pth')
    optimizer_path = os.path.join(save_dir, 'optimizer.pth')
    network.save(network_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    print('[OPTIMIZER] save: %s' % optimizer_path)
    print('')


def run_testers(tester_dict, framework, test_data_loader, test_dir):
    for key, tester in tester_dict.items():
        tester_dir = os.path.join(test_dir, key)
        tester.run(framework, test_data_loader, tester_dir)
        print('[TEST] %s: %s' % (key, tester_dir))
    print('')


def update_learning_rate(optimizer, decay_rate):
    decay_dict = dict()
    for param_group in optimizer.param_groups:
        pre_lr = param_group['lr']
        param_group['lr'] = pre_lr * decay_rate
        decay_dict[pre_lr] = param_group['lr']

    decay_str = cvt_dict2str(decay_dict)
    decay_str = decay_str.replace(' ', ' | ').replace(':', ' -> ')
    print('[OPTIMIZER] learning rate: %s' % decay_str)
    print('')


def update_network(network, optimizer, loss_dict, max_grad=None):
    sum_loss = 0
    for _, loss in loss_dict.items():
        sum_loss += loss
    optimizer.zero_grad()
    sum_loss.backward()
    if max_grad is not None:
        nn.utils.clip_grad_norm_(network.parameters(), max_grad)
    optimizer.step()
