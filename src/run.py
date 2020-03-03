import os
import time
import torch
import traceback
from tensorboardX import SummaryWriter
import option
import util


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('[RUN] parse arguments')
    args, framework, optimizer, data_loader_dict, tester_dict = option.parse_options()

    print('[RUN] create result directories')
    result_dir_dict = util.create_result_dir(args.result_dir, ['src', 'log', 'snapshot', 'test'])
    util.copy_file(args.bash_file, args.result_dir)
    util.copy_dir('./src', result_dir_dict['src'])

    print('[RUN] create loggers')
    train_log_dir = os.path.join(result_dir_dict['log'], 'train')
    train_logger = SummaryWriter(train_log_dir)

    print('[OPTIMIZER] learning rate:', optimizer.param_groups[0]['lr'])
    n_batches = data_loader_dict['train'].__len__()
    global_step = args.training_args['init_iter']

    print('')
    skip_flag = False
    while True:
        start_time = time.time()
        for train_data_dict in data_loader_dict['train']:
            batch_time = time.time() - start_time

            if skip_flag:
                skip_flag = False
            else:
                if global_step in args.snapshot_iters:
                    snapshot_dir = os.path.join(result_dir_dict['snapshot'], '%07d' % global_step)
                    util.save_snapshot(framework.network, optimizer, snapshot_dir)

                if global_step in args.test_iters:
                    test_dir = os.path.join(result_dir_dict['test'], '%07d' % global_step)
                    util.run_testers(tester_dict, framework, data_loader_dict['test'], test_dir)

                if args.training_args['max_iter'] <= global_step:
                    break

                if global_step in args.training_args['lr_decay_schd'].keys():
                    util.update_learning_rate(optimizer, args.training_args['lr_decay_schd'][global_step])

            train_loss_dict, train_time = \
                train_network_one_step(args, framework, optimizer, train_data_dict, global_step)

            if train_loss_dict is None:
                skip_flag = True
                train_data_dict.clear()
                del train_data_dict

            else:
                if global_step % args.training_args['print_intv'] == 0:
                    iter_str = '[TRAINING] %d/%d:' % (global_step, args.training_args['max_iter'])
                    info_str = 'n_batches: %d, batch_time: %0.3f, train_time: %0.3f' % \
                               (n_batches, batch_time, train_time)
                    train_str = util.cvt_dict2str(train_loss_dict)
                    print(iter_str + '\n- ' + info_str + '\n- ' + train_str + '\n')

                    for key, value in train_loss_dict.items():
                        train_logger.add_scalar(key, value, global_step)

                train_loss_dict.clear()
                train_data_dict.clear()
                del train_loss_dict, train_data_dict
                global_step += 1

            start_time = time.time()
        if args.training_args['max_iter'] <= global_step:
            break
    train_logger.close()


def train_network_one_step(args, framework, optimizer, train_data_dict, global_step):
    train_loss_dict = None
    try:
        start_time = time.time()
        _, train_loss_dict = framework.train_forward(train_data_dict)
        util.update_network(framework.network, optimizer, train_loss_dict, args.training_args['max_grad'])
        train_time = time.time() - start_time

    except Exception as e:
        print('[WARNING] %d/%d:' % (global_step, args.training_args['max_iter']))
        print(traceback.print_exc())
        print('- %s\n- skip this mini-batch\n' % (str(e)))

        if 'memory' in str(e):
            optimizer.zero_grad()
            optimizer.step()
            torch.cuda.empty_cache()

        if train_loss_dict is not None:
            train_loss_dict.clear()
            del train_loss_dict
        return None, None
    return train_loss_dict, train_time


if __name__ == '__main__':
    main()
