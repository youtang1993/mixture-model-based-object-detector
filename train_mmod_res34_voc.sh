#!/usr/bin/env bash

# CUDA_DEVICE_ORDER=PCI_BUS_ID
# CUDA_VISIBLE_DEVICES=0,1,2,3
BASH_FILE="./train_standard_mmod_voc.sh"
RESULT_DIR="./result/mmod/voc/`(date "+%Y%m%d%H%M%S")`-320x320-mmod_res34"

python3 ./src/run_mmod.py \
--bash_file=${BASH_FILE} \
--result_dir=${RESULT_DIR} \
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
    'lw_dict': {'mog_nll': 1.0, 'sample_comb_nll': 2.0},
    'n_samples': 5
}" \
--post_proc_args="{
    'pi_thresh': 0.001, 'conf_thresh': 0.001,
    'nms_thresh': 0.5, 'max_boxes': 1000
}" \
--optimizer_args="{
    'lr': 0.003, 'momentum': 0.9, 'weight_decay': 0.0005
}" \
--train_data_set_info="{
    'dataset': 'voc',
    'args': {
        'roots': ['./data/voc-devkit-2007/VOC2007', './data/voc-devkit-2012/VOC2012'],
        'types': ['trainval', 'trainval'],
        'pre_proc': '', 'preproc_args': {'max_boxes': 100,
            'rgb_mean': [0.485, 0.456, 0.406],
            'rgb_std': [0.229, 0.224, 0.225]}
    }
}" \
--test_data_set_info="{
    'dataset': 'voc',
    'args': {
        'roots': ['./data/voc-devkit-2007/VOC2007'],
        'types': ['test', 'test'],
        'pre_proc' '', 'preproc_args': {'max_boxes': 100,
            'rgb_mean': [0.485, 0.456, 0.406],
            'rgb_std': [0.229, 0.224, 0.225]}
    }
}" \
--tester_info_list="[{
    'tester': 'image',
    'args': {
        'n_samples': 50, 'conf_thresh': 0.2,
        'max_boxes': 20, 'result_dir': $RESULT_DIR/image
    }
}, {
    'tester': 'quant',
    'args': {
        'dataset': 'voc_2007_test',
        'result_dir': $RESULT_DIR/quant
    }
}]" \
\
--train_args="{
    'init_iter': 0, 'max_iter': 0, 'max_grad': 7, 'print_intv': 100,
    'lr_decay_schd': '{40000: 0.1, 70000: 0.1}'
}" \
--test_iters="[40000, 70000, 100000]" \
--snapshot_iters="[40000, 70000, 100000]" \
\
# --load_dir='enter snaphot-dir-path'
