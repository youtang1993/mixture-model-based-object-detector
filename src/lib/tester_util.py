import numpy as np
import cv2

colors = ((64, 64, 64), (31, 119, 180), (174, 199, 232), (255, 127, 14),
          (255, 187, 120), (44, 160, 44), (152, 223, 138), (214, 39, 40),
          (255, 152, 150), (148, 103, 189), (197, 176, 213), (140, 86, 75),
          (196, 156, 148), (227, 119, 194), (247, 182, 210), (127, 127, 127),
          (199, 199, 199), (188, 189, 34), (219, 219, 141), (23, 190, 207),
          (158, 218, 229), (180, 119, 31))


def draw_boxes(img_s, boxes_s, confs_s=None, labels_s=None,
               class_map=None, conf_thresh=0.0, max_boxes=100):

    box_img_s = img_s.copy()
    n_draw_boxes = 0
    n_wrong_boxes = 0
    n_thresh_boxes = 0
    for i, box in enumerate(boxes_s):
        try:
            l, t = int(round(box[0])), int(round(box[1]))
            r, b = int(round(box[2])), int(round(box[3]))
        except IndexError:
            print(boxes_s)
            print(i, box)
            print('IndexError')
            exit()

        if confs_s is not None:
            if conf_thresh > confs_s[i]:
                n_thresh_boxes += 1
                continue
        if (r - l <= 0) or (b - t <= 0):
            n_wrong_boxes += 1
            continue
        if n_draw_boxes >= max_boxes:
            continue

        conf_str = '-' if confs_s is None else '%0.3f' % confs_s[i]
        if labels_s is None:
            lab_str, color = '-', colors[i % len(colors)]
        else:
            lab_i = int(labels_s[i])
            lab_str = str(lab_i) if class_map is None else class_map[lab_i]
            color = colors[lab_i % len(colors)]

        box_img_s = cv2.rectangle(box_img_s, (l, t), (r, b), color, 2)
        l = int(l - 1 if l > 1 else r - 60)
        t = int(t - 8 if t > 8 else b)
        r, b = int(l + 60), int(t + 8)
        box_img_s = cv2.rectangle(box_img_s, (l, t), (r, b), color, cv2.FILLED)
        box_img_s = cv2.putText(box_img_s, '%s %s' % (conf_str, lab_str), (l + 1, t + 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255),
                                1, cv2.LINE_AA)
        n_draw_boxes += 1

    info_text = 'n_draw_b: %d, n_thr_b: %d, n_wrong_b: %d' % \
                (n_draw_boxes, n_thresh_boxes, n_wrong_boxes)
    if confs_s is not None:
        info_text += ', sum_of_conf: %.3f' % (np.sum(confs_s))
    else:
        info_text += ', sum_of_conf: -'

    box_img_s = cv2.rectangle(box_img_s, (0, 0), (350, 11), (0, 0, 0), cv2.FILLED)
    box_img_s = cv2.putText(box_img_s, info_text, (5, 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.25, (255, 255, 255), 1, cv2.LINE_AA)
    return box_img_s
