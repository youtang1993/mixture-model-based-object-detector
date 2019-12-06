import abc
import torch.nn as nn
import torchvision
from .network_util import init_modules_xavier


def get_backbone_dict():
    return {
        'res34fpn': ResNet34FPN,
        'res50fpn': ResNet50FPN,
        'res101fpn': ResNet101FPN
    }


class BackboneABC(nn.Module):
    def __init__(self, network_args):
        super(BackboneABC, self).__init__()
        self.pretrained = network_args['pretrained']
        self.net = nn.ModuleDict()

    @ abc.abstractmethod
    def build(self):
        pass

    @ abc.abstractmethod
    def get_fmap2img_ratios(self):
        pass


class ResNet34FPN(BackboneABC):
    def __init__(self, network_args):
        super(ResNet34FPN, self).__init__(network_args)
        self.inter_chs = (128, 256, 512)
        self.fmap_ch = network_args['fmap_ch']

    def __get_base_network__(self):
        return torchvision.models.resnet34(pretrained=self.pretrained)

    def get_fmap2img_ratios(self):
        return 1/8.0, 1/16.0, 1/32.0, 1/64.0, 1/128.0

    def build(self):
        base_net = self.__get_base_network__()
        self.net['base'] = nn.Sequential(
            base_net.conv1, base_net.bn1, base_net.relu,
            base_net.maxpool, base_net.layer1)

        self.net['stage_c3'] = base_net.layer2
        self.net['stage_c4'] = base_net.layer3
        self.net['stage_c5'] = base_net.layer4
        self.net['stage_c6'] = nn.Conv2d(self.inter_chs[2], self.fmap_ch, 3, 2, 1)
        self.net['stage_c7'] = nn.Sequential(
            nn.ReLU(), nn.Conv2d(self.fmap_ch, self.fmap_ch, 3, 2, 1))

        self.net['stage_p3'] = nn.Conv2d(self.inter_chs[0], self.fmap_ch, 1, 1, 0)
        self.net['stage_p4'] = nn.Conv2d(self.inter_chs[1], self.fmap_ch, 1, 1, 0)
        self.net['stage_p5'] = nn.Conv2d(self.inter_chs[2], self.fmap_ch, 1, 1, 0)
        self.net['stage_p6'] = nn.Conv2d(self.fmap_ch, self.fmap_ch, 1, 1, 0)
        self.net['stage_p7'] = nn.Conv2d(self.fmap_ch, self.fmap_ch, 1, 1, 0)

        if self.pretrained:
            print('[Backbone]: load pretrained model')
        else:
            init_modules_xavier(
                [self.net['base'], self.net['stage_c3'], self.net['stage_c4'],
                 self.net['stage_c5']])
        init_modules_xavier(
            [self.net['stage_c6'], self.net['stage_c7'], self.net['stage_p3'],
             self.net['stage_p4'], self.net['stage_p5'], self.net['stage_p6'],
             self.net['stage_p7']])

    def forward(self, image):
        base_fmap = self.net['base'].forward(image)
        fmap_c3 = self.net['stage_c3'].forward(base_fmap)
        fmap_c4 = self.net['stage_c4'].forward(fmap_c3)
        fmap_c5 = self.net['stage_c5'].forward(fmap_c4)
        fmap_c6 = self.net['stage_c6'].forward(fmap_c5)
        fmap_c7 = self.net['stage_c7'].forward(fmap_c6)

        fmap_p3 = self.net['stage_p3'].forward(fmap_c3)
        fmap_p4 = self.net['stage_p4'].forward(fmap_c4)
        fmap_p5 = self.net['stage_p5'].forward(fmap_c5)
        fmap_p6 = self.net['stage_p6'].forward(fmap_c6)
        fmap_p7 = self.net['stage_p7'].forward(fmap_c7)
        return [fmap_p3, fmap_p4, fmap_p5, fmap_p6, fmap_p7]


class ResNet50FPN(ResNet34FPN):
    def __init__(self, network_args):
        super(ResNet34FPN, self).__init__(network_args)
        self.inter_chs = (512, 1024, 2048)

    def __get_base_network__(self):
        return torchvision.models.resnet50(pretrained=self.pretrained)


class ResNet101FPN(ResNet50FPN):
    def __get_base_network__(self):
        return torchvision.models.resnet101(pretrained=self.pretrained)
