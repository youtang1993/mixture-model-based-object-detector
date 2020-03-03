import os
import abc
import traceback
import numpy as np
import scipy.misc
from shutil import copyfile
from xml.etree import ElementTree
from torch.utils.data.dataset import Dataset
from .pre_proc import get_pre_proc_dict
from lib.external.dataset.roidb import combined_roidb


def get_dataset_dict():
    return {
        'voc': VOCDataset,
        'coco': COCODataset,
    }


class DatasetABC(abc.ABC, Dataset):
    def __init__(self, global_args, dataset_args):
        super(DatasetABC, self).__init__()
        self.global_args = global_args
        self.roots = dataset_args['roots']
        self.types = dataset_args['types']
        pre_proc_class = get_pre_proc_dict()[dataset_args['pre_proc']]
        self.pre_proc = pre_proc_class(global_args, dataset_args['pre_proc_args'])

    @ abc.abstractmethod
    def get_name2number_map(self):
        pass

    @ abc.abstractmethod
    def get_number2name_map(self):
        pass

    @ abc.abstractmethod
    def get_dataset_roots(self):
        pass


class VOCDataset(DatasetABC):
    def __init__(self, global_args, dataset_args):
        super(VOCDataset, self).__init__(global_args, dataset_args)

        img_pathes = list()
        anno_pathes = list()
        for root_dir, set_type in zip(self.roots, self.types):
            set_path = os.path.join(root_dir, 'ImageSets', 'Main', '%s.txt' % set_type)
            img_path_form = os.path.join(root_dir, 'JPEGImages', '%s.jpg')
            anno_path_form = os.path.join(root_dir, 'Annotations', '%s.xml')

            with open(set_path) as file:
                for img_name in file.readlines():
                    img_name = img_name.strip('\n')
                    img_pathes.append(img_path_form % img_name)
                    anno_pathes.append(anno_path_form % img_name)

        self.img_pathes = np.array(img_pathes).astype(np.string_)
        self.anno_pathes = np.array(anno_pathes).astype(np.string_)

        self.name2number_map = {
            'background': 0,
            'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4,
            'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9,
            'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13,
            'motorbike': 14, 'person': 15, 'pottedplant': 16,
            'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}
        self.number2name_map = {
            0: 'background',
            1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
            5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

    def __len__(self):
        return len(self.img_pathes)

    def __getitem__(self, data_idx):
        img = scipy.misc.imread(self.img_pathes[data_idx])
        anno = ElementTree.parse(self.anno_pathes[data_idx]).getroot()
        boxes, labels = self.__parse_anno__(anno)

        sample_dict = {'img': img, 'boxes': boxes, 'labels': labels}
        try:
            sample_dict = self.pre_proc.process(sample_dict)
        except Exception:
            print(traceback.print_exc())
            print('- %s\n' % self.img_pathes[data_idx])
            copyfile( self.img_pathes[data_idx].replace('coco2017', 'coco2017-2'), self.img_pathes[data_idx])
            sample_dict = self.__getitem__(data_idx)
        return sample_dict

    def __getitem_tmp__(self, data_idx):
        img = scipy.misc.imread(self.img_pathes[data_idx])
        anno = ElementTree.parse(self.anno_pathes[data_idx]).getroot()
        boxes, labels = self.__parse_anno__(anno)

        sample_dict = {'img': img, 'boxes': boxes, 'labels': labels}
        sample_dict = self.pre_proc.process(sample_dict)
        return sample_dict

    def __parse_anno__(self, anno):
        boxes = list()
        labels = list()
        for obj in anno.findall('object'):
            bndbox = obj.find('bndbox')
            boxes.append([
                float(bndbox.find('xmin').text), float(bndbox.find('ymin').text),
                float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)])
            labels.append(np.array(self.name2number_map[obj.find('name').text]))
        boxes = np.array(boxes)
        labels = np.array(labels)
        return boxes, labels

    def get_name2number_map(self):
        return self.name2number_map

    def get_number2name_map(self):
        return self.number2name_map

    def get_dataset_roots(self):
        return self.roots


class COCODataset(DatasetABC):
    def __init__(self, global_args, dataset_args):
        super(COCODataset, self).__init__(global_args, dataset_args)

        imdb_names = "coco_2017_" + dataset_args['types'][0]
        # imdb_names = "coco_2017_" + type.replace('_', '-')
        # print(imdb_names, self.roots[0])
        # exit()
        self.roidb = combined_roidb(imdb_names, self.roots[0])
        self.data_size = len(self.roidb)

        self.number2name_map = {
            0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
            5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench', 15: 'bird',
            16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant',
            22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack', 26: 'umbrella',
            27: 'handbag', 28: 'tie', 29: 'suitcase', 30: 'frisbee', 31: 'skis', 32: 'snowboard',
            33: 'sports ball', 34: 'kite', 35: 'baseball bat', 36: 'baseball glove',
            37: 'skateboard', 38: 'surfboard', 39: 'tennis racket', 40: 'bottle',
            41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife', 45: 'spoon', 46: 'bowl',
            47: 'banana', 48: 'apple', 49: 'sandwich', 50: 'orange', 51: 'broccoli',
            52: 'carrot', 53: 'hot dog', 54: 'pizza', 55: 'donut', 56: 'cake',
            57: 'chair', 58: 'couch', 59: 'potted plant', 60: 'bed', 61: 'dining table',
            62: 'toilet', 63: 'tv', 64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard',
            68: 'cell phone', 69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink',
            73: 'refrigerator', 74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors',
            78: 'teddy bear', 79: 'hair drier', 80: 'toothbrush'}

        self.name2number_map = {
            'background': 0, 'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4,
            'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10,
            'fire hydrant': 11, 'stop sign': 12, 'parking meter': 13, 'bench': 14, 'bird': 15,
            'cat': 16, 'dog': 17, 'horse': 18, 'sheep': 19, 'cow': 20, 'elephant': 21, 'bear': 22,
            'zebra': 23, 'giraffe': 24, 'backpack': 25, 'umbrella': 26, 'handbag': 27,
            'tie': 28, 'suitcase': 29, 'frisbee': 30, 'skis': 31, 'snowboard': 32,
            'sports ball': 33, 'kite': 34, 'baseball bat': 35, 'baseball glove': 36,
            'skateboard': 37, 'surfboard': 38, 'tennis racket': 39, 'bottle': 40,
            'wine glass': 41, 'cup': 42, 'fork': 43, 'knife': 44, 'spoon': 45, 'bowl': 46,
            'banana': 47, 'apple': 48, 'sandwich': 49, 'orange': 50, 'broccoli': 51,
            'carrot': 52, 'hot dog': 53, 'pizza': 54, 'donut': 55, 'cake': 56,
            'chair': 57, 'couch': 58, 'potted plant': 59, 'bed': 60, 'dining table': 61,
            'toilet': 62, 'tv': 63, 'laptop': 64, 'mouse': 65, 'remote': 66, 'keyboard': 67,
            'cell phone': 68, 'microwave': 69, 'oven': 70, 'toaster': 71, 'sink': 72,
            'refrigerator': 73, 'book': 74, 'clock': 75, 'vase': 76, 'scissors': 77,
            'teddy bear': 78, 'hair drier': 79, 'toothbrush': 80}

    def __getitem__(self, index):
        try:
            minibatch_db = self.roidb[index]
            img = scipy.misc.imread(minibatch_db['image'])

            if len(img.shape) == 2:
                img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
            boxes = minibatch_db['boxes']
            labels = minibatch_db['gt_classes']

            sample_dict = {'img': img, 'boxes': boxes, 'labels': labels}
            sample_dict = self.pre_proc.process(sample_dict)
        except Exception:
            print(traceback.print_exc())
            print('- %s\n' % minibatch_db['image'])
            copyfile(minibatch_db['image'].replace('coco2017', 'coco2017-2'), minibatch_db['image'])
            sample_dict = self.__getitem__(index)
        return sample_dict

    def __len__(self):
        return len(self.roidb)

    def get_number2name_map(self):
        return self.number2name_map

    def get_name2number_map(self):
        return self.name2number_map

    def get_dataset_roots(self):
        return self.roots

