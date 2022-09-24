import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from utils.metrics import ap_per_class, coco_eval, get_batch_statistics
from utils.plots import plot_images
from utils.torch_utils import import_fun, is_parallel, time_synchronized
from utils.yolo_utils import non_max_suppression, output_to_target


def init_model(opt):

    # from opt.data load "nc"、"names" config
    filter_cls = opt.filter_cls
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    num_classes = int(data_dict['nc'])  # number of classes
    class_names = data_dict['filter_names'] if filter_cls else data_dict[
        'names']  # class names
    # check num_classes and class_names
    assert len(class_names
               ) == num_classes, '%g names found for nc=%g dataset in %s' % (
                   len(class_names), num_classes, opt.data)

    data_dict['name'] = str(Path(opt.data).stem)  # dataset name

    # Model parameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    # load model
    model = import_fun("models", opt.model.strip())(class_names, hyp,
                                                    opt.verbose)
    model = model.to(opt.device)

    return model, data_dict, hyp


def get_optimizer(hyp, model, freeze_layers=[], verbose=False):
    # Freeze
    if len(freeze_layers):
        print("\n freeze layers:", end="")
        freeze_layers = [f'{x}.' for x in freeze_layers]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze_layers):
                v.requires_grad = False
                if verbose:
                    print(f'freezing {k}')
    model.model_info(verbose)

    if hyp['optim_mode'] == "all":
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        if hyp['optimizer'] == "SGD":
            optimizer = optim.SGD(parameters,
                                  lr=hyp['lr0'],
                                  momentum=hyp['momentum'],
                                  weight_decay=hyp['weight_decay'],
                                  nesterov=hyp.get('nesterov', True))
        else:
            optimizer = optim.Adam(parameters,
                                   lr=hyp['lr0'],
                                   betas=(hyp['momentum'], 0.999))
        del parameters
    else:
        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(
                    v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)

        g0 = list(filter(lambda p: p.requires_grad, g0))
        g1 = list(filter(lambda p: p.requires_grad, g1))
        g2 = list(filter(lambda p: p.requires_grad, g2))

        if hyp['optimizer'] == "Adam":
            optimizer = optim.Adam(g0,
                                   lr=hyp['lr0'],
                                   betas=(hyp['momentum'],
                                          0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(g0,
                                  lr=hyp['lr0'],
                                  momentum=hyp['momentum'],
                                  nesterov=hyp.get('nesterov', True))

        optimizer.add_param_group({
            'params': g1,
            'weight_decay': hyp['weight_decay']
        })  # add g1 with weight_decay
        optimizer.add_param_group({'params': g2})  # add g2 (biases)
        print(
            f"{'optimizer:'} {type(optimizer).__name__} with parameter groups: "
            + f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
        del g0, g1, g2

    return optimizer


def load_checkpoint(checkpoint, opt, model, optimizer, lr_scheduler,
                    results_file, epochs, device, accumulate, num_batches):
    start_epoch, best_fitness = 0, 0.0

    if checkpoint.strip() != '':
        print(
            "--------------------------------------Resume--------------------------------------\n"
        )
        ckpt = torch.load(checkpoint, map_location=device)  # load checkpoint
        # model state_dict
        model._load_state_dict_(ckpt['model'])
        
        # if not resume training from checkpoint,
        # only load pretrained state_dict for finetune
        if opt.resume:
            # epoch
            start_epoch = ckpt['epoch'] + 1
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (
                checkpoint, epochs)

            if epochs < start_epoch:
                print("checkpoint '%s' has been trained for %g epochs" %
                      (checkpoint, ckpt['epoch']))
            else:
                print("checkpoint '%s' have trained at epoch %g " %
                      (checkpoint, start_epoch - 1))

            # optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # lr_scheduler
            # note: WarmupCosineLR在batch iter内step
            lr_scheduler.last_epoch = (start_epoch -
                                       1) * num_batches // accumulate

            # training_results
            if ckpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(ckpt['training_results'])  # write results.txt
        print(
            "----------------------------------------------------------------------------------\n"
        )
        del ckpt
    return start_epoch, best_fitness


@torch.no_grad()
def evaluate(model,
             dataloader,
             batch_size,
             device,
             tb_writer=None,
             half=False,
             eval_on_training=False,
             conf_thres=0.001,
             nms_thres=0.6,
             compute_yolo_loss=None,
             save_json=False,
             is_coco=False,
             run_dir="./",
             verbose=True):
    # get num_classes、class_names
    num_classes, class_names = (
        model.module.num_classes,
        model.module.class_names) if is_parallel(model) else (
            model.num_classes, model.class_names)

    img_size = dataloader.dataset.img_size  # get image size from dataset
    save_json &= is_coco  # save json only for coco dataset

    # variables for evaluate
    loss = torch.zeros(3, device=device)
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R',
                                 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    seen = 0
    json_dict, stats, ap, ap_class = [], [], [], []

    # Half
    half = device.type != 'cpu' and half  # half precision only supported on CUDA
    if half:
        model.half()
    model.eval()  # eval mode

    # run once for warmup if not training
    if not eval_on_training:
        for i in range(10):
            model(
                torch.zeros(1, 3, img_size[1], img_size[0]).to(device).type_as(
                    next(model.parameters())))

    # start evaluate
    for batch_i, (imgs, targets, paths,
                  shapes0) in enumerate(tqdm(dataloader, desc=s)):

        t = time_synchronized()
        imgs = imgs.to(device, non_blocking=True) / 255.0
        imgs = imgs.half() if half else imgs.float()  # uint8 to fp16/32

        targets = targets.to(device)

        t0 += time_synchronized() - t

        # Run model
        t = time_synchronized()
        inf_out, train_out = model(imgs)  # inference and training outputs
        t1 += time_synchronized() - t

        # Compute loss
        if eval_on_training:
            loss += compute_yolo_loss(
                [x.float() for x in train_out],
                targets,
            )[1]  # box, obj, cls

        # Run NMS
        t = time_synchronized()
        output = non_max_suppression(inf_out, conf_thres, nms_thres)
        t2 += time_synchronized() - t

        # Statistics per batch
        seen, stats, json_dict = get_batch_statistics(imgs, targets.clone(),
                                                      paths, shapes0, output,
                                                      seen, stats, save_json,
                                                      json_dict, is_coco)

        # Plot images
        if batch_i < 3 and verbose:
            f = Path(run_dir) / ('test_batch%g_gt.jpg' % batch_i)
            # plot ground truth
            plot_images(imgs, targets, paths, str(f), class_names)
            f = Path(run_dir) / ('test_batch%g_pred.jpg' % batch_i)
            # plot predictions
            plot_images(imgs, output_to_target(output, imgs.shape[2:4]), paths,
                        str(f), class_names)

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(
            *stats,
            plot=verbose,
            save_dir=run_dir,
            names=class_names,
        )
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # number of targets per class
        nt = np.bincount(stats[3].astype(np.int64), minlength=num_classes)
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.4g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and num_classes > 1 and num_classes <= 20 and not eval_on_training and len(
            stats):
        for i, c in enumerate(ap_class):
            print(pf %
                  (class_names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    times = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
    shape = (batch_size, 3, img_size[0], img_size[1])
    print(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}'
        % times)

    # Save JSON and coco mAP eval
    if save_json and len(json_dict):
        # write results to json file
        pred_json_file = str(
            Path(run_dir) / 'detections_{}_yolov4_results.json'.format(
                Path(dataloader.dataset.path).stem))  # must str for coco_eval

        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json_file)
        with open(pred_json_file, 'w') as file:
            json.dump(json_dict, file)

        # coco mAP eval
        gt_json_file = dataloader.dataset.gt_json_file
        # image IDs to evaluate
        imgIds = [
            int(Path(x).stem) if Path(x).stem.isnumeric() else int(
                Path(x).stem.split('_')[-1])
            for x in dataloader.dataset.img_files
        ]
        map, map50 = coco_eval(gt_json_file, pred_json_file, imgIds)

    # mAP per class
    maps = np.zeros(num_classes) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    return (mp, mr, map50, map,
            *(loss.cpu() / len(dataloader)).tolist()), maps, times
