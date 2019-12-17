import os
import abc
import sys
import tqdm
import pickle
import numpy as np
import scipy.misc
from lib import util
from lib.external.dataset import factory
from . import tester_util


def get_tester_dict():
    return {
        'image': ImageTester,
        'quant': QuantTester,
    }


class TesterABC(abc.ABC):
    def __init__(self, global_args, tester_args):
        self.global_args = global_args
        self.tester_args = tester_args

    @ abc.abstractmethod
    def run(self, framework, data_loader, result_dir):
        pass


class ImageTester(TesterABC):
    def __init__(self, global_args, tester_args):
        super(ImageTester, self).__init__(global_args, tester_args)
        self.n_samples = tester_args['n_samples']
        self.max_boxes = tester_args['max_boxes']
        self.conf_thresh = tester_args['conf_thresh']

    def run(self, framework, data_loader, result_dir):
        assert data_loader.batch_size == 1
        pre_proc = data_loader.dataset.pre_proc
        class_map = data_loader.dataset.get_number2name_map()
        util.make_dir(result_dir)

        for i, data_dict in enumerate(data_loader):
            if i >= self.n_samples:
                break

            output_dict, result_dict = framework.infer_forward(data_dict)
            pred_boxes_s = util.cvt_torch2numpy(result_dict['boxes_l'])[0]
            pred_confs_s = util.cvt_torch2numpy(result_dict['confs_l'])[0]
            pred_labels_s = util.cvt_torch2numpy(result_dict['labels_l'])[0]

            data_dict = pre_proc.inv_transform_batch(data_dict)
            img_s = data_dict['img'][0]
            gt_boxes_s = data_dict['boxes'][0]
            gt_labels_s = data_dict['labels'][0]

            sort_idx = 0
            gt_img_path = os.path.join(result_dir, '%03d_%d_%s.png' % (i, sort_idx, 'gt'))
            gt_img_s = tester_util.draw_boxes(
                img_s, gt_boxes_s, None, gt_labels_s,
                class_map, self.conf_thresh, self.max_boxes)
            scipy.misc.imsave(gt_img_path, gt_img_s)
            sort_idx += 1

            # draw_boxes
            pred_img_path = os.path.join(result_dir, '%03d_%d_%s.png' % (i, sort_idx, 'pred'))
            pred_img_s = tester_util.draw_boxes(
                img_s, pred_boxes_s, pred_confs_s, pred_labels_s,
                class_map, self.conf_thresh, self.max_boxes)
            scipy.misc.imsave(pred_img_path, pred_img_s)


class QuantTester(TesterABC):
    def __init__(self, global_args, tester_args):
        super(QuantTester, self).__init__(global_args, tester_args)
        self.n_classes = global_args['n_classes']
        self.imdb_name = tester_args['dataset']
        assert self.imdb_name in ('voc_2007_test', 'coco_2017_val', 'coco_2017_test-dev')

    def run(self, framework, data_loader, result_dir):
        assert data_loader.batch_size == 1
        num_samples = data_loader.dataset.__len__()
        all_boxes = [[[] for _ in range(num_samples)] for _ in range(self.n_classes)]

        times = list()
        data_loader_pbar = tqdm.tqdm(data_loader)
        for idx, data_dict in enumerate(data_loader_pbar):

            output_dict, result_dict = framework.infer_forward(data_dict)
            times.append(result_dict['time'])
            data_loader_pbar.set_description('infer time: %.4f sec' % np.mean(times))

            # total predict boxes shape : (batch, # pred box, 4)
            # total predict boxes confidence shape : (batch, # pred box, 1)
            # total predict boxes label shape : (batch, # pred box, 1)
            img_size_s = data_dict['img_size'].float()[0]
            input_size = data_dict['img'].shape[2:]

            boxes_s = result_dict['boxes_l'][0]
            confs_s = result_dict['confs_l'][0]
            labels_s = result_dict['labels_l'][0]

            boxes_s[:, [0, 2]] *= (img_size_s[1] / input_size[1])
            boxes_s[:, [1, 3]] *= (img_size_s[0] / input_size[0])
            boxes_s, confs_s, labels_s = \
                util.sort_boxes_s(boxes_s, confs_s, labels_s)

            boxes_s = util.cvt_torch2numpy(boxes_s)
            confs_s = util.cvt_torch2numpy(confs_s)
            labels_s = util.cvt_torch2numpy(labels_s)

            if len(confs_s.shape) == 1:
                confs_s = np.expand_dims(confs_s, axis=1)

            for i, (cls_box, cls_conf, cls_label) in \
                    enumerate(zip(boxes_s, confs_s, labels_s)):
                cls_box_with_conf = np.concatenate((cls_box, cls_conf), axis=0)
                cls_box_with_conf = np.expand_dims(cls_box_with_conf, axis=0)
                all_boxes[int(cls_label)][idx].append(cls_box_with_conf)

            for c in range(self.n_classes):
                all_boxes[c][idx] = np.concatenate(all_boxes[c][idx], axis=0) \
                    if 0 < len(all_boxes[c][idx]) else np.concatenate([[]], axis=0)
            data_dict.clear()

        # create result directories
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        dataset_root = data_loader.dataset.get_dataset_roots()[0]
        imdb = factory.get_imdb(self.imdb_name, dataset_root)

        if 'coco' in self.imdb_name:
            sys_stdout = sys.stdout
            result_file_path = open(os.path.join(result_dir, 'ap_ar.txt'), 'w')
            sys.stdout = result_file_path
            imdb.evaluate_detections(all_boxes, result_dir)
            sys.stdout = sys_stdout
            result_file_path.close()

        else:
            det_file_path = os.path.join(result_dir, 'detection_results.pkl')
            with open(det_file_path, 'wb') as det_file:
                pickle.dump(all_boxes, det_file, pickle.HIGHEST_PROTOCOL)
            result_msg = imdb.evaluate_detections(all_boxes, result_dir)
            result_file_path = os.path.join(result_dir, 'mean_ap.txt')
            with open(result_file_path, 'w') as file:
                file.write(result_msg)
            os.remove(det_file_path)

        all_boxes.clear()
