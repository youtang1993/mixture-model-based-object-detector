import abc
from . import loss_func_util as loss_util


class LossFunctionABC(abc.ABC):
    def __init__(self, global_args, loss_args):
        self.global_args = global_args
        self.loss_args = loss_args

    @ abc.abstractmethod
    def forward(self, *x):
        pass


class MMODLossFunction(LossFunctionABC):
    def __init__(self, global_args, loss_func_args):
        super(MMODLossFunction, self).__init__(global_args, loss_func_args)
        self.lw_dict = loss_func_args['lw_dict']
        self.n_samples = loss_func_args['n_samples']
        self.n_classes = global_args['n_classes']
        assert 'mog_nll' in self.lw_dict.keys()
        assert 'mm_nll' in self.lw_dict.keys()

    def forward(self, mu, sig, prob, pi, boxes, labels, n_boxes):
        mog_nll_loss = loss_util.calc_mog_nll(mu, sig, pi, boxes, n_boxes)
        mm_nll_loss = loss_util.calc_mm_nll(
            mu.detach(), sig.detach(), pi.detach(), prob,
            boxes, labels, n_boxes, self.n_samples, self.n_classes)

        mog_nll_loss = self.lw_dict['mog_nll'] * mog_nll_loss
        mm_nll_loss = self.lw_dict['mm_nll'] * mm_nll_loss
        return mog_nll_loss, mm_nll_loss
