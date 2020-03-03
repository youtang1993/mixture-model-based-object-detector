import torch
from . import util as lib_util

'''
def __calc_xywh_nll__(mu, sig, pi, boxes, n_boxes):
    xywh_nll_loss = torch.zeros(1).cuda()
    for i in range(mu.shape[0]):
        if n_boxes[i] <= 0:
            xywh_nll_loss += torch.zeros(1).cuda()
        else:
            boxes_s = boxes[i:i + 1, :n_boxes[i]]
            mixture_lhs_s = util.calc_mog_likelihood(
                mu[i:i + 1], sig[i:i + 1], pi[i:i + 1], boxes_s, sum_gauss=True)

            mixture_lhs_s *= n_boxes[i]
            mixture_nll_s = -torch.log(mixture_lhs_s + epsilon)
            xywh_nll_loss += torch.sum(mixture_nll_s)

    # xywh_nll_loss /= torch.sum(n_boxes)
    return xywh_nll_loss
'''


def calc_mog_nll(mu, sig, pi, boxes, n_boxes):
    # print(torch.min(mu), torch.mean(mu), torch.max(mu))
    # print(torch.min(sig), torch.mean(sig), torch.max(sig))
    # print(torch.min(pi), torch.mean(pi), torch.max(pi))
    mog_nll_loss = torch.zeros(1).cuda()
    for i in range(mu.shape[0]):
        if n_boxes[i] <= 0:
            mog_nll_loss += torch.zeros(1).cuda()
        else:
            mu_s, sig_s, pi_s = mu[i:i + 1], sig[i:i + 1], pi[i:i + 1]
            boxes_s = boxes[i:i + 1, :n_boxes[i]]

            mixture_lhs_s = lib_util.mog_pdf(mu_s, sig_s, pi_s, boxes_s, sum_gauss=True)
            mixture_lhs_s *= n_boxes[i]
            mixture_nll_s = -torch.log(mixture_lhs_s + lib_util.epsilon)

            # print(torch.min(mixture_nll_s), torch.mean(mixture_nll_s), torch.max(mixture_nll_s))
            mog_nll_loss += torch.sum(mixture_nll_s)

    return mog_nll_loss


def calc_mod_mm_nll(mu, sig, pi, clsprob, boxes, labels, n_boxes, n_samples, n_classes):
    bg_labels_s = torch.zeros((torch.max(n_boxes) * n_samples, n_classes)).float().cuda()
    bg_labels_s[:, 0] = 1.0

    labels = lib_util.cvt_int2onehot(labels, n_classes)
    sample_boxes = lib_util.sample_coords_from_mog(mu, sig, pi, int(torch.max(n_boxes) * n_samples))

    # sample_boxes = list()
    # sample_boxes.append(torch.ones((32, int(torch.max(n_boxes) * n_samples), 1)).float().cuda() * 2)
    # sample_boxes.append(torch.ones((32, int(torch.max(n_boxes) * n_samples), 1)).float().cuda() * 3)
    # sample_boxes.append(torch.ones((32, int(torch.max(n_boxes) * n_samples), 1)).float().cuda() * 4)
    # sample_boxes.append(torch.ones((32, int(torch.max(n_boxes) * n_samples), 1)).float().cuda() * 5)
    # sample_boxes = torch.cat(sample_boxes, dim=2)

    mod_mm_nll_loss = torch.zeros(1).cuda()
    for i in range(mu.shape[0]):
        mu_s, sig_s = mu[i:i + 1], sig[i:i + 1]
        pi_s, clsprob_s = pi[i:i + 1], clsprob[i:i + 1]
        boxes_s = boxes[i:i + 1, :n_boxes[i]]
        labels_s = labels[i:i + 1, :n_boxes[i]]
        sample_boxes_s = sample_boxes[i:i + 1, :n_boxes[i] * n_samples]

        if n_boxes[i] <= 0:
            sample_labels_s = bg_labels_s[:n_boxes[i] * n_samples]
        else:
            iou_pairs = lib_util.calc_jaccard_torch(sample_boxes_s[0], boxes_s[0])
            max_ious, argmax_ious = torch.max(iou_pairs, dim=1)
            sample_labels_s = labels_s[0, argmax_ious]
            sample_labels_s = torch.where(
                max_ious.unsqueeze(dim=1) > 0.5, sample_labels_s,
                bg_labels_s[:n_boxes[i] * n_samples])
        sample_labels_s = sample_labels_s.unsqueeze(dim=0)

        gauss_lhs_s = lib_util.mog_pdf(mu_s, sig_s, pi_s, sample_boxes_s, sum_gauss=False)
        gauss_lhs_s *= n_boxes[i]
        cat_probs_s = lib_util.category_pmf(clsprob_s, sample_labels_s.float())

        mm_lhs_s = torch.sum(gauss_lhs_s * cat_probs_s, dim=3)
        mm_nll_s = -torch.log(mm_lhs_s + lib_util.epsilon)
        mod_mm_nll_loss += torch.sum(mm_nll_s)
    return mod_mm_nll_loss
