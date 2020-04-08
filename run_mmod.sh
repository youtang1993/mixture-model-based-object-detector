#!/usr/bin/env bash

python3 ./src/run.py \
--bash_file="./run_mmod.sh" \
--result_dir="./result/`(date "+%Y%m%d%H%M%S")`-320x320-res50fpn" \
\
--global_args="{
    'n_classes': 81, 'batch_size': 32,
    'img_h': 320, 'img_w': 320,
    'coord_h': 10, 'coord_w': 10,
    'devices': [0], 'main_device': 0,
}" \
--network_args="{
    'pretrained': True, 'backbone': 'res50fpn', 'fmap_ch': 256,
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
    'lr': 0.005, 'momentum': 0.9, 'weight_decay': 0.00005
}" \
--train_data_loader_info="{
    'dataset': 'coco',
    'dataset_args': {
        'roots': ['./data/coco2017'],
        'types': ['train'],
        'pre_proc': 'augm', 'pre_proc_args': {
            'max_boxes': 300,
            'rgb_mean': [0.485, 0.456, 0.406],
            'rgb_std': [0.229, 0.224, 0.225]}
    },
    'shuffle': True, 'num_workers': 4
}" \
--test_data_loader_info="{
    'dataset': 'coco',
    'dataset_args': {
        'roots': ['./data/coco2017'],
        'types': ['val'],
        'pre_proc': 'base', 'pre_proc_args': {
            'max_boxes': 300,
            'rgb_mean': [0.485, 0.456, 0.406],
            'rgb_std': [0.229, 0.224, 0.225]}
    },
    'shuffle': False, 'num_workers': 2, 'batch_size': 1
}" \
--tester_info_list="[{
    'tester': 'image',
    'tester_args': {'n_samples': 50, 'conf_thresh': 0.2, 'max_boxes': 20}
}, {
    'tester': 'quant',
    'tester_args': {'dataset': 'coco_2017_val'}
}]" \
--training_args="{
    'init_iter': 0, 'max_iter': 500000, 'max_grad': 7, 'print_intv': 100,
    'lr_decay_schd': {350000: 0.1, 430000: 0.1, 470000: 0.1}
}" \
--test_iters="[20000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 480000, 490000, 495000, 500000]" \
--snapshot_iters="[50000, 100000, 150000, 200000, 300000, 400000, 450000, 480000, 490000, 500000]" \
\
--load_dir="None"
