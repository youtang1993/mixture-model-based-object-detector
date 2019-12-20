import abc
import numpy as np
from . import util as lib_util
from . import pre_proc_util as pre_util


def get_pre_proc_dict():
    return {
        'base': PreProcBase,
        'augm': PreProcAugm,
    }


class PreProcABC(abc.ABC):
    def __init__(self, global_args, pre_proc_args):
        self.n_classes = global_args['n_classes']
        self.input_size = (global_args['img_h'], global_args['img_w'])
        self.coord_range = (global_args['coord_h'], global_args['coord_w'])
        self.max_boxes = pre_proc_args['max_boxes']

    def __fill__(self, sample_dict):
        def create_dummy_boxes(_n_dummies):
            boxes = list()
            labels = list()
            for _ in range(_n_dummies):
                boxes.append(np.array([0, 0, 0, 0]))
                labels.append(np.array([0]))
            return np.array(boxes), np.array(labels)

        n_boxes = sample_dict['boxes'].shape[0]
        n_dummies = self.max_boxes - n_boxes

        if n_dummies > 0:
            dummy_boxes, dummy_labels = create_dummy_boxes(n_dummies)
            sample_dict['boxes'] = np.concatenate((sample_dict['boxes'], dummy_boxes), axis=0)
            sample_dict['boxes'] = sample_dict['boxes'].astype(np.float32)
            if 'labels' in sample_dict.keys():
                sample_dict['labels'] = np.concatenate((sample_dict['labels'], dummy_labels), axis=0)
                sample_dict['labels'] = sample_dict['labels'].astype(np.float32)
        else:
            sample_dict['boxes'] = sample_dict['boxes'][:self.max_boxes]
            sample_dict['boxes'] = sample_dict['boxes'].astype(np.float32)
            if 'labels' in sample_dict.keys():
                sample_dict['labels'] = sample_dict['labels'][:self.max_boxes]
                sample_dict['labels'] = sample_dict['labels'].astype(np.float32)
        return sample_dict

    @ abc.abstractmethod
    def __augment__(self, sample_dict):
        pass

    @ abc.abstractmethod
    def transform(self, sample_dict):
        pass

    @ abc.abstractmethod
    def inv_transform_batch(self, data_dict):
        pass

    @ abc.abstractmethod
    def process(self, sample_dict):
        pass


class PreProcBase(PreProcABC):
    def __init__(self, global_args, pre_proc_args):
        super(PreProcBase, self).__init__(global_args, pre_proc_args)
        self.rgb_mean = np.array(pre_proc_args['rgb_mean']).astype(np.float32).reshape(3, 1, 1)
        self.rgb_std = np.array(pre_proc_args['rgb_std']).astype(np.float32).reshape(3, 1, 1)

    def __augment__(self, sample_dict):
        return sample_dict

    def transform(self, sample_dict):
        s_dict = sample_dict
        s_dict['img'] = np.transpose(s_dict['img'], axes=(2, 0, 1)).astype(dtype=np.float32) / 255.0
        s_dict['img'] = (s_dict['img'] - self.rgb_mean) / self.rgb_std
        s_dict['boxes'][:, [0, 2]] *= (self.coord_range[1] / self.input_size[1])
        s_dict['boxes'][:, [1, 3]] *= (self.coord_range[0] / self.input_size[0])
        s_dict['labels'] = np.expand_dims(s_dict['labels'], axis=1)
        return s_dict

    def inv_transform_batch(self, data_dict):
        d_dict = lib_util.cvt_torch2numpy(data_dict)
        d_dict['img'] = d_dict['img'] * self.rgb_std + self.rgb_mean
        d_dict['img'] = (np.transpose(d_dict['img'], axes=(0, 2, 3, 1)) * 255.0).astype(dtype=np.uint8)
        d_dict['boxes'][:, :, [0, 2]] *= (self.input_size[1] / self.coord_range[1])
        d_dict['boxes'][:, :, [1, 3]] *= (self.input_size[0] / self.coord_range[0])
        d_dict['labels'] = np.squeeze(d_dict['labels'], axis=2)
        return d_dict

    def process(self, sample_dict):
        sample_dict['img'] = np.array(sample_dict['img']).astype(np.float32)
        sample_dict['boxes'] = np.array(sample_dict['boxes']).astype(np.float32)
        sample_dict['labels'] = np.array(sample_dict['labels']).astype(np.float32)

        s_dict = self.__augment__(sample_dict)
        img_size = np.array(s_dict['img'].shape)[:2]
        s_dict['img'], s_dict['boxes'] = pre_util.resize(s_dict['img'], s_dict['boxes'], self.input_size)
        s_dict['boxes'] = lib_util.clip_boxes_s(s_dict['boxes'], self.input_size, numpy=True)

        n_boxes = s_dict['boxes'].shape[0]
        s_dict = self.transform(s_dict)
        s_dict.update({'n_boxes': n_boxes, 'img_size': img_size})
        s_dict = self.__fill__(s_dict)
        return s_dict


class PreProcAugm(PreProcBase):
    def __augment__(self, sample_dict):
        img = np.array(sample_dict['img'])
        boxes = np.array(sample_dict['boxes'])
        labels = np.array(sample_dict['labels'])

        img = pre_util.rand_brightness(img)
        img = pre_util.rand_contrast(img)
        img, boxes = pre_util.expand(img, boxes)
        img, boxes, labels = pre_util.rand_crop(img, boxes, labels)
        img, boxes = pre_util.rand_flip(img, boxes)

        sample_dict['img'] = img
        sample_dict['boxes'] = boxes
        sample_dict['labels'] = labels
        return sample_dict
