from pathlib import Path

import numpy as np
import torch

from utils.plots import plot_mc_curve, plot_pr_curve
from utils.yolo_utils import (box_iou, scale_coords, xywh2xyxy, xyxy2xywh)


def yolo_fitness(x, is_coco):
    # Returns fitness for best.pt

    # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    w = [0.0, 0.0, 0.1, 0.9] if is_coco else [0.0, 0.0, 1.0, 0.0]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp,
                 conf,
                 pred_cls,
                 target_cls,
                 plot=False,
                 save_dir='.',
                 names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros(
        (nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0],
                              left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0],
                              left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:,
                                                                           j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec,
                                        mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px,
                      f1,
                      Path(save_dir) / 'F1_curve.png',
                      names,
                      ylabel='F1')
        plot_mc_curve(px,
                      p,
                      Path(save_dir) / 'P_curve.png',
                      names,
                      ylabel='Precision')
        plot_mc_curve(px,
                      r,
                      Path(save_dir) / 'R_curve.png',
                      names,
                      ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    # i = r.mean(0).argmax()  # max Recall index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(
            mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def coco80_to_coco91_class():
    # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    x = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90
    ]
    return x


def get_batch_statistics(imgs, targets, paths, shapes0, output, seen, stats,
                         save_json, json_dict, is_coco):
    """ Compute true positives, predicted scores and predicted labels per sample """
    device = targets.device
    
    _, _, height, width = imgs.shape  # batch size, channels, height, width
    targets[:, 2:] *= torch.Tensor([width, height, width,
                                    height]).to(device)  # to pixels
    # iou vector for mAP@0.5:0.95
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()  # number of iou (10)

    # Statistics per image
    for si, pred in enumerate(output):
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        path, shape0 = Path(paths[si]), shapes0[si]
        seen += 1

        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool),
                              torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Predictions
        predn = pred.clone()
        predn[:, :4] = scale_coords(imgs[si].shape[1:], predn[:, :4],
                                    shape0)  # native-space pred

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            tbox = scale_coords(imgs[si].shape[1:], tbox,
                                shape0)  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox),
                                1)  # native-space labels
            correct = process_batch(predn, labelsn, iouv)
        else:
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(),
                      tcls))  # (correct, conf, pcls, tcls)

        if save_json:
            # append to COCO-JSON dictionary
            json_dict = save_one_json(predn, json_dict, path, is_coco)

    return seen, stats, json_dict


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls,
                                                     *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, is_coco):
    # Append to pycocotools JSON dictionary
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    image_id = int(path.stem) if path.stem.isnumeric() else int(
        path.stem.split('_')[-1])
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)
        })
    return jdict


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0],
                          iouv.shape[0],
                          dtype=torch.bool,
                          device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    # IoU above threshold and classes match
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                            1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def coco_eval(gt_json_file, pred_json_file, imgIds):
    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    from utils.pycocotools.coco import COCO
    from utils.pycocotools.cocoeval import COCOeval
    # init gt
    cocoGt = COCO(gt_json_file)
    # initialize COCO pred api
    cocoDt = cocoGt.loadRes(pred_json_file)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    # image IDs to evaluate
    cocoEval.params.imgIds = imgIds  # only eval these imgs
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    # get results (mAP@0.5:0.95, mAP@0.5)
    map, map50 = cocoEval.stats[:2]

    # return
    return map, map50
