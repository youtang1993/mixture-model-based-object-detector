import torch
from torchvision.ops.boxes import nms
from . import util as lib_util


class MMODPostProc(object):
    def __init__(self, global_args, post_proc_args):
        self.input_size = (global_args['img_h'], global_args['img_w'])
        self.coord_range = (global_args['coord_h'], global_args['coord_w'])
        self.n_classes = global_args['n_classes']

        self.pi_thresh = post_proc_args['pi_thresh']
        self.conf_thresh = post_proc_args['conf_thresh']
        self.nms_thresh = post_proc_args['nms_thresh']

    def __filter_cls_boxes_s__(self, boxes_s, confs_s, pi_s):
        boxes_sl = list()
        confs_sl = list()
        labels_sl = list()

        norm_pi_s = pi_s / torch.max(pi_s)
        keep_idxes = torch.nonzero(norm_pi_s > self.pi_thresh).view(-1)
        boxes_s = boxes_s[:, keep_idxes]
        confs_s = confs_s[:, keep_idxes]

        for c in range(self.n_classes - 1):
            boxes_sc = boxes_s[c]
            confs_sc = confs_s[c]

            if len(boxes_sc) == 0:
                continue

            keep_idxes = torch.nonzero(confs_sc > self.conf_thresh).view(-1)
            boxes_sc = boxes_sc[keep_idxes]
            confs_sc = confs_sc[keep_idxes]
            if keep_idxes.shape[0] == 0:
                continue

            if self.nms_thresh <= 0.0:
                boxes_sc, confs_sc = lib_util.sort_boxes_s(boxes_sc, confs_sc)
                boxes_sc, confs_sc = boxes_sc[:1], confs_sc[:1]
            elif self.nms_thresh > 1.0:
                pass
            else:
                keep_idxes = nms(boxes_sc, confs_sc, self.nms_thresh)
                keep_idxes = keep_idxes.long().view(-1)
                boxes_sc = boxes_sc[keep_idxes]
                confs_sc = confs_sc[keep_idxes].unsqueeze(dim=1)

            labels_css = torch.zeros(confs_sc.shape).float().cuda()
            labels_css += c

            boxes_sl.append(boxes_sc)
            confs_sl.append(confs_sc)
            labels_sl.append(labels_css)

        if len(boxes_sl) > 0:
            boxes_s = torch.cat(boxes_sl, dim=0)
            confs_s = torch.cat(confs_sl, dim=0)
            labels_s = torch.cat(labels_sl, dim=0)
        else:
            boxes_s = torch.zeros((1, 4)).float().cuda()
            confs_s = torch.zeros((1, 1)).float().cuda()
            labels_s = torch.zeros((1, 1)).float().cuda()
        return boxes_s, confs_s, labels_s

    def forward(self, mu, prob, pi):
        boxes = mu.transpose(1, 2).clone()
        boxes[:, :, [0, 2]] = boxes[:, :, [0, 2]] * (self.input_size[1] / self.coord_range[1])
        boxes[:, :, [1, 3]] = boxes[:, :, [1, 3]] * (self.input_size[0] / self.coord_range[0])
        boxes = lib_util.clip_boxes(boxes, self.input_size)
        boxes = torch.cat([boxes.unsqueeze(dim=1)] * (self.n_classes - 1), dim=1)
        confs = prob[:, 1:]

        boxes_l, confs_l, labels_l = list(), list(), list()
        for i, (boxes_s, confs_s) in enumerate(zip(boxes, confs)):
            boxes_s, confs_s, labels_s = self.__filter_cls_boxes_s__(boxes_s, confs_s, pi[i, 0])
            boxes_l.append(boxes_s)
            confs_l.append(confs_s)
            labels_l.append(labels_s + 1)
        return boxes_l, confs_l, labels_l
