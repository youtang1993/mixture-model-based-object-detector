import os
import abc
import numpy as np
import scipy.misc
from lib import util


class TesterABC(abc.ABC):
    def __init__(self, global_args, tester_args):
        self.global_args = global_args
        self.tester_args = tester_args
        self.result_root = tester_args['result_root']

    @ abc.abstractmethod
    def test(self, framework, data_loader, result_dir):
        pass


class ImageTester(TesterABC):
    def __init__(self, global_args, tester_args):
        super(ImageTester, self).__init__(global_args, tester_args)
        self.n_samples = tester_args['n_samples']
        self.max_boxes = tester_args['max_boxes']
        self.conf_thresh = tester_args['conf_thresh']
        self.coord_size = (global_args['coord_h'], global_args['coord_w'])

    def test(self, framework, data_loader, result_dir):
        assert data_loader.batch_size == 1
        pre_proc = data_loader.dataset.pre_proc
        class_map = data_loader.dataset.get_number2name_map()
        save_dir = os.path.join(self.result_root, result_dir)
        util.make_dir(save_dir)

        for i, data_dict in enumerate(data_loader):
            if i >= self.n_samples:
                break

            output_dict, result_dict = framework.infer_forward(data_dict)
            pred_boxes_s = util.cvt_torch2numpy(result_dict['boxes_l'])[0]
            pred_confs_s = util.cvt_torch2numpy(result_dict['confs_l'])[0]
            pred_labels_s = util.cvt_torch2numpy(result_dict['labels_l'])[0]

            data_dict = pre_proc.inv_process(data_dict)
            img_s = data_dict['img'][0]
            gt_boxes_s = data_dict['boxes'][0]
            gt_labels_s = data_dict['labels'][0]
            img_size = data_dict['img_size'][0]
            input_size = img_s.shape[:2]

            sort_idx = 0
            gt_img_path = os.path.join(save_dir, '%03d_%d_%s.png' % (i, sort_idx, 'gt'))
            gt_img_s = testutil.draw_boxes(img_s, gt_boxes_s, None, gt_labels_s,
                                           class_map, self.conf_thresh, self.max_boxes)
            scipy.misc.imsave(gt_img_path, gt_img_s)
            del gt_img_s

            data_dict.clear()
            output_dict.clear()
            result_dict.clear()
        return save_dir


def save_img(save_dir, img_dict, key, size, range_ratio, idx, sort_idx, tag):
    if key in img_dict.keys():
        img = util.cvt_torch2numpy(img_dict[key])[0] * range_ratio
        if len(img.shape) == 3:
            img = np.squeeze(np.transpose(img, (1, 2, 0)))
        img = scipy.misc.imresize(img, size, interp='bilinear')
        img_path = os.path.join(save_dir, '%03d_%d_%s.png' % (idx, sort_idx, tag))
        scipy.misc.imsave(img_path, img)
    return sort_idx + 1
