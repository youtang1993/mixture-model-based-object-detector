import abc
import time
import torch
import torch.nn as nn
from .external.sync_batchnorm import convert_model, patch_replication_callback


class FrameworkABC(abc.ABC):
    def __init__(self, global_args, network, post_proc):
        self.global_args = global_args
        self.post_proc = post_proc
        self.network = network

    def train_forward(self, data_dict):
        loss_dict = self.forward(data_dict, train=True, grad_enable=True)
        return {}, loss_dict

    def valid_forward(self, data_dict):
        loss_dict = self.forward(data_dict, train=True, grad_enable=False)
        return {}, loss_dict

    def infer_forward(self, data_dict):
        output_dict, result_dict = self.forward(data_dict, train=False, grad_enable=False)
        return output_dict, result_dict

    @ abc.abstractmethod
    def forward(self, data_dict, train=True, grad_enable=True):
        pass


class MMODFramework(FrameworkABC):
    def __init__(self, global_args, network, post_proc):
        super(MMODFramework, self).__init__(global_args, network, post_proc)
        devices = global_args['devices']
        self.main_device = global_args['main_device']

        if len(devices) > 1:
            network = convert_model(network)
            self.network = nn.DataParallel(
                network, device_ids=devices, output_device=self.main_device)
            patch_replication_callback(self.network)
        self.network.cuda(self.main_device)

    def forward(self, data_dict, train=True, grad_enable=True):
        self.network.train(train)
        torch.autograd.set_grad_enabled(grad_enable)

        with torch.cuda.device(self.main_device):
            image = data_dict['img'].requires_grad_(grad_enable).float().cuda()
            if train:
                boxes = data_dict['boxes'].cuda()
                labels = data_dict['labels'].long().cuda()
                n_boxes = data_dict['n_boxes'].long().cuda()
                mog_nll_loss, sample_comb_nll_loss = \
                    self.network.forward(image, boxes, labels, n_boxes, loss=True)

                sum_n_boxes = torch.sum(n_boxes)
                mog_nll_loss = torch.sum(mog_nll_loss) / sum_n_boxes
                sample_comb_nll_loss = \
                    torch.sum(sample_comb_nll_loss) / (sum_n_boxes * self.network.loss_func.n_samples)
                return {'mog_nll': mog_nll_loss,
                        'sample_comb_nll': sample_comb_nll_loss}

            else:
                s_t = time.time()
                mu, sig, prob, pi = self.network.forward(image, loss=False)
                boxes_l, confs_l, labels_l = self.post_proc.forward(mu, sig, prob, pi)
                inf_time = time.time() - s_t
                return {'mu': mu, 'sig': sig, 'prob': prob, 'pi': pi}, \
                       {'boxes_l': boxes_l, 'confs_l': confs_l, 'labels_l': labels_l, 'time': inf_time}
