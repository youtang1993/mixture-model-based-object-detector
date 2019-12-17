#!/usr/bin/env bash

python3 ./src/run.py \
--bash_file="./train_mmod_res34_voc.sh" \
--result_dir="./result/`(date "+%Y%m%d%H%M%S")`-voc-320x320-mmod_res34" \
\
--global_args="{
    'n_classes': 21, 'batch_size': 32,
    'img_h': 320, 'img_w': 320,
    'coord_h': 10, 'coord_w': 10,
    'devices': [0], 'main_device': 0,
}" \
--network_args="{
    'pretrained': True, 'backbone': 'res34fpn', 'fmap_ch': 256,
    'xy_limit_factor': 1.0, 'std_factor': 0.1
}" \
--loss_func_args="{
    'lw_dict': {'mog_nll': 1.0, 'mod_nll': 2.0},
    'n_samples': 5
}" \
--post_proc_args="{
    'pi_thresh': 0.001, 'conf_thresh': 0.001,
    'nms_thresh': 0.5, 'max_boxes': 1000
}" \
--optimizer_args="{
    'lr': 0.003, 'momentum': 0.9, 'weight_decay': 0.0005
}" \
--train_data_loader_info="{
    'dataset': 'voc',
    'dataset_args': {
        'roots': ['./data/voc-devkit-2007/VOC2007', './data/voc-devkit-2012/VOC2012'],
        'types': ['trainval', 'trainval'],
        'pre_proc': 'augm', 'pre_proc_args': {
            'max_boxes': 100,
            'rgb_mean': [0.485, 0.456, 0.406],
            'rgb_std': [0.229, 0.224, 0.225]}
    },
    'shuffle': True, 'num_workers': 6
}" \
--test_data_loader_info="{
    'dataset': 'voc',
    'dataset_args': {
        'roots': ['./data/voc-devkit-2007/VOC2007'],
        'types': ['test'],
        'pre_proc': 'base', 'pre_proc_args': {
            'max_boxes': 100,
            'rgb_mean': [0.485, 0.456, 0.406],
            'rgb_std': [0.229, 0.224, 0.225]}
    },
    'num_workers': 1
}" \
--tester_info_list="[{
    'tester': 'image',
    'tester_args': {
        'n_samples': 50, 'conf_thresh': 0.2,
        'max_boxes': 20, 'result_root': $RESULT_DIR/image
    }
}, {
    'tester': 'quant',
    'tester_args': {
        'dataset': 'voc_2007_test',
        'result_root': $RESULT_DIR/quant
    }
}]" \
\
--training_args="{
    'init_iter': 0, 'max_iter': 100000, 'max_grad': 7, 'print_intv': 100,
    'lr_decay_schd': {40000: 0.1, 70000: 0.1}
}" \
--test_iters="[40000, 70000, 90000, 95000, 100000]" \
--snapshot_iters="[40000, 70000, 100000]" \
\
#--load_dir=""
