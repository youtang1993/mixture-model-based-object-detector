import os
import time
import torch
import traceback
from tensorboardX import SummaryWriter
import util as run_util
import option


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    args, framework, optimizer, data_loader_dict, tester_dict = option.parse_options()

    print('CREATE RESULT DIRECTORIES')
    result_dir_dict = run_util.create_result_dir(args.result_dir, ['src', 'log', 'snapshot'])
    run_util.copy_file(args.bash_file, args.result_dir)
    run_util.copy_dir('./src', result_dir_dict['src'])

    print('CREATE LOGGERS')
    train_log_dir = os.path.join(result_dir_dict['log'], 'train')
    train_logger = SummaryWriter(train_log_dir)

    print('')
    print('START TRAINING')
    print('[OPTIMIZER] learning rate:', optimizer.param_groups[0]['lr'])
    n_batches = data_loader_dict['train'].__len__()
    global_step = args.train_info_dict['init_iter']
    max_grad = args.train_info_dict['max_grad']
    skip_flag = False

    while True:
        start_time = time.time()
        for train_data_dict in data_loader_dict['train']:
            batch_time = time.time() - start_time

            try:
                if not skip_flag:
                    if global_step in args.snapshot_iters:
                        snapshot_dir = os.path.join(result_dir_dict['snapshot'], '%07d' % global_step)
                        run_util.save_snapshot(framework.network, optimizer, snapshot_dir)

                    # if global_step in args.test_iters:
                    #     snapshot_dir = os.path.join(result_dir_dict['snapshot'], '%07d' % global_step)
                    #     test_networks(args, tester_dict, framework, test_set_loader, global_step)

                    if args.train_info_dict['max_iter'] <= global_step:
                        break

                    # update_learning_rate(optimizer, args.lr_decay_schd_dict, global_step)
                    # update_loss_weights({'loss_func': loss_func}, args.lw_schd_dict, global_step)

                try:
                    start_time = time.time()
                    _, train_loss_dict = framework.train_forward(train_data_dict)
                    run.update_networks(network, optimizer, train_loss_dict, max_grad)
                    train_time = time.time() - start_time

                    # if valid_loss_dict is not None:
                    if global_step % args.train_info_dict['print_intv'] == 0:
                        iter_str = '[%d/%d] ' % (global_step, args.train_info_dict['max_iter'])
                        info_str = 'n_batches:%d batch_time:%0.3f train_time:%0.3f' % \
                                   (n_batches, batch_time, train_time)
                        train_str = util.cvt_dict2str(train_loss_dict)

                        print(iter_str + info_str)
                        print('[train] ' + train_str + '\n')

                        for key, value in train_loss_dict.items():
                            train_logger.add_scalar(key, value, global_step)

                    train_loss_dict.clear()
                    del train_loss_dict

                    skip_flag = False
                    global_step += 1

                except Exception as e:
                    print('[WARNING] %s' % (str(e)))
                    print('[%d/%d] skip this mini-batch\n' % (global_step, args.train_info_dict['max_iter']))

                    if 'memory' in str(e):
                        optimizer.zero_grad()
                        optimizer.step()

                        train_loss_dict.clear()
                        del train_loss_dict
                        torch.cuda.empty_cache()

                    skip_flag = True

                train_data_dict.clear()
                start_time = time.time()

            except Exception as e:
                traceback.print_tb(e.__traceback__)
                print('[ERROR] %s' % (str(e)))
                snapshot_dir = os.path.join(result_dir_dict['snapshot'], '%07d' % global_step)
                run_util.save_snapshot(framework.network, optimizer, snapshot_dir)
                exit()
    train_logger.close()


if __name__ == '__main__':
    main()
