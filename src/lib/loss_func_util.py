import torch
from . import util as lib_util


def calc_mog_nll(mu, sig, pi, boxes, n_boxes):
    mog_nll_loss = list()
    # mog_nll_loss = torch.zeros(1).cuda()
    for i in range(mu.shape[0]):
        if n_boxes[i] <= 0:
            mog_nll_loss.append(torch.zeros(1).cuda())
        else:
            mu_s, sig_s, pi_s = mu[i:i + 1], sig[i:i + 1], pi[i:i + 1]
            boxes_s = boxes[i:i + 1, :n_boxes[i]]

            mixture_lhs_s = lib_util.mog_pdf(mu_s, sig_s, pi_s, boxes_s, sum_gauss=True)
            mixture_lhs_s *= n_boxes[i]
            mixture_nll_s = -torch.log(mixture_lhs_s + lib_util.epsilon)
            mog_nll_loss.append(mixture_nll_s.squeeze(dim=0).squeeze(dim=1))
    mog_nll_loss = torch.cat(mog_nll_loss, dim=0)
    return mog_nll_loss


def calc_mod_mm_nll(mu, sig, pi, clsprob, boxes, labels, n_boxes, n_samples, n_classes):
    bg_labels_s = torch.zeros((torch.max(n_boxes) * n_samples, n_classes)).float().cuda()
    bg_labels_s[:, 0] = 1.0

    labels = lib_util.cvt_int2onehot(labels, n_classes)
    sample_boxes = lib_util.sample_coords_from_mog(mu, sig, pi, int(torch.max(n_boxes) * n_samples))

    mod_mm_nll_loss = list()
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

        # gauss_lhs_s = util.calc_mog_likelihood(mu_s, sig_s, pi_s, samples_s, sum_gauss=False)
        gauss_lhs_s = lib_util.mog_pdf(mu_s, sig_s, pi_s, sample_boxes_s, sum_gauss=False)
        gauss_lhs_s *= n_boxes[i]
        cat_probs_s = lib_util.category_pmf(clsprob_s, sample_labels_s.float())

        mm_lhs_s = torch.sum(gauss_lhs_s * cat_probs_s, dim=3)
        mm_nll_s = -torch.log(mm_lhs_s + lib_util.epsilon)
        mod_mm_nll_loss.append(mm_nll_s.squeeze(dim=0).squeeze(dim=1))
    mod_mm_nll_loss = torch.cat(mod_mm_nll_loss, dim=0)
    return mod_mm_nll_loss
