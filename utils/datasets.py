import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.augmentations import data_augments
from utils.general import get_hash
from utils.yolo_utils import xywhn2xyxy

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']


def create_dataloader(data_dict,
                      img_size,
                      batch_size,
                      augment=False,
                      hyp=None,
                      task="val",
                      mode="normal",
                      multi_scale=False,
                      filter_cls=False):
    if task != "train":
        augment, hyp = False, None

    dataset = yolo_dataset(
        data_dict=data_dict,
        img_size=img_size,
        batch_size=batch_size,
        multi_scale=multi_scale,  # multi_scale training
        augment=augment,  # augment images
        hyp=hyp,  # data augment hyp
        task=task,  # train/val/test
        mode=mode,
        filter_cls=filter_cls)

    batch_size = min(batch_size, len(dataset))
    # number of workers
    nw = min([os.cpu_count(), batch_size, 8])

    print('%s dataloader using %g workers' % (task, nw))

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=(task != "val"),
                            num_workers=nw,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn,
                            worker_init_fn=dataset.worker_init_fn)

    return dataloader


class yolo_dataset(Dataset):  # for training/testing

    def __init__(self,
                 data_dict,
                 task="val",
                 img_size=[640, 640],
                 batch_size=16,
                 augment=False,
                 hyp=None,
                 filter_cls=False,
                 mode="normal",
                 multi_scale=False):
        # init dataset config
        assert mode in ["normal", "train_paired"]

        self.multi_scale = multi_scale
        self.task = task
        self.mode = mode
        self.path = data_dict[task]
        self.name = data_dict['name']

        if mode == "train_paired":
            # assert multi_scale == False
            self.low_train_path = data_dict["s2t_train"]

        # gt json file for coco eval api
        self.gt_json_file = data_dict[
            "annotation"] if "annotation" in data_dict.keys() else None

        self.img_size = img_size  # w, h
        self.batch_size = batch_size
        self.augment = augment
        self.filter_cls = filter_cls
        self.batch_count = -1
        self.hyp = hyp

        if self.multi_scale:
            assert task not in ["val", "test"]
            self.img_size = [512, 512]  # 512 for check_anchors

        # load image and label files
        self.load_img_and_label_files()
        # load and check labels
        self.check_labels()

        # filter labels for specific class
        if self.filter_cls:
            self.labels = filter_labels_for_specific_classes(
                labels=self.labels,
                names=data_dict['names'],
                filter_names=data_dict['filter_names'])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        mode = self.mode
        img_file_path = self.img_files[index]
        # Load image
        # shapes(hw_original) for scale_coords when eval
        img, shapes = load_image(img_file_path, self.img_size)
        # Load labels
        labels = self.labels[index].copy()

        if mode == "train_paired":
            low_img_file_path = self.low_img_files[index]

            assert os.path.basename(low_img_file_path) == os.path.basename(
                img_file_path)  # paired

            low_img, _ = load_image(low_img_file_path, self.img_size)
            img = [img, low_img]
        else:
            img = [img]
        # data augment
        if self.augment:
            img, labels = data_augments(img, labels, hyp=self.hyp)

        if mode == "train_paired":
            gt_pixel_mask, gt_feature_mask = build_gt_mask(
                labels, img[0].shape[0:2])

        nL = len(labels)  # number of labels
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # return
        if mode == "train_paired":
            low_img = img2tensor(img[1])
            img = img2tensor(img[0])
            return img, low_img, labels_out, labels_out.clone(
            ), img_file_path, low_img_file_path, shapes, shapes, gt_pixel_mask, gt_feature_mask
        else:
            img = img2tensor(img[0])
            return img, labels_out, img_file_path, shapes

    def collate_fn(self, batch):
        # multi_scale training : selects new image size every 10 batch
        if self.multi_scale:
            self.batch_count += 1
            if self.batch_count % 10 == 0:
                # img_size: 320:32:608
                img_size = random.randint(10, 19) * 32  # update img_size
                self.img_size = [img_size, img_size]  # update img_size

        if self.mode == "train_paired":
            imgs, low_imgs, labels, low_labels, paths, low_paths, shapes, low_shapes, gt_pixel_mask, gt_feature_mask = zip(
                *batch)

            low_imgs = torch.cat(low_imgs, 0)
            imgs = torch.cat(imgs, 0)
            imgs = torch.cat([imgs, low_imgs], 0)

            labels = labels + low_labels
            for i, l in enumerate(labels):
                l[:, 0] = i  # add target image index for build_targets()
            labels = torch.cat(labels, 0)

            paths = paths + low_paths
            shapes = shapes + low_shapes

            gt_pixel_mask = torch.cat(gt_pixel_mask, 0)
            gt_feature_mask = torch.cat(gt_feature_mask, 0)

            return imgs, labels, paths, shapes, gt_pixel_mask, gt_feature_mask
        else:

            imgs, labels, paths, shapes = zip(*batch)
            imgs = torch.cat(imgs, 0)
            for i, l in enumerate(labels):
                l[:, 0] = i  # add target image index for build_targets()
            labels = torch.cat(labels, 0)

            return imgs, labels, paths, shapes

    def worker_init_fn(self, worker_id):
        # See for details of numpy:
        # https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
        # See for details of random:
        # https://pytorch.org/docs/stable/notes/randomness.html#dataloader

        # NumPy
        uint64_seed = torch.initial_seed()
        ss = np.random.SeedSequence([uint64_seed])
        np.random.seed(ss.generate_state(4))
        # random
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)

    def load_img_and_label_files(self):
        self.img_files = load_img_files_from_txt(self.path)
        self.label_files = img2label_paths(self.img_files)

        if self.mode == "train_paired":
            self.low_img_files = load_img_files_from_txt(self.low_train_path)

        self.n = len(self.img_files)  # number of images
        assert self.n > 0, 'No images found in %s.\n' % (self.path)

    def check_labels(self, ):
        # Check cache
        cache_path = str(Path(self.label_files[0]).parent.parent
                         ) + '/' + self.name + '.%s.cache' % self.task

        try:
            cache = torch.load(cache_path)
            assert cache['hash'] == get_hash(self.label_files +
                                             self.img_files)  # dataset changed
        except Exception:
            cache = self.cache_labels(cache_path)  # cache

        # Get labels
        labels, shapes = zip(*[cache[img_file] for img_file in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)

        # Check labels and img_files
        try:
            nm, nf, ne, nd = 0, 0, 0, 0  # number missing, found, empty, duplicate
            pbar = tqdm(self.label_files)
            for i, file in enumerate(pbar):
                l = self.labels[i]  # label
                if l.shape[0]:
                    # [class, x, y, w, h]
                    assert l.shape[1] == 5, '> 5 label columns: %s' % file
                    assert (l >= 0).all(), 'negative labels: %s' % file
                    assert (l[:, 1:] <= 1).all() and (l[:, 1:] >= 0).all(
                    ), 'non-normalized or out of bounds coordinate labels: %s' % file
                    # duplicate rows
                    if np.unique(l, axis=0).shape[0] < l.shape[0]:
                        nd += 1
                    self.labels[i] = l
                    nf += 1  # file found
                else:
                    ne += 1

                pbar.desc = 'Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                    cache_path, nf, nm, ne, nd, self.n)

            assert nf, "Can not train without labels. \n"
            assert (nf + ne) == self.n, "num(labels) != num(images) \n"

        except Exception:
            cache = self.cache_labels(cache_path)  # cache

    def cache_labels(self, path='labels.cache'):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files),
                    desc='Scanning images',
                    total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                shape = cv2.imread(img).shape[:2]  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array(
                            [x.split() for x in f.read().splitlines()],
                            dtype=np.float32)  # labels
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                x[img] = None
                print('WARNING: %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x


def filter_labels_for_specific_classes(labels, names, filter_names):
    for i, labels_ in enumerate(
            tqdm(labels, desc="filter labels for specific class detection")):
        remain_labels = []
        for label in labels_:
            # filter
            class_name = names[int(label[0])]
            if class_name in filter_names:
                label[0] = filter_names.index(class_name)
                remain_labels.append(label)
        remain_labels = np.asarray(remain_labels)
        if len(remain_labels) == 0:
            remain_labels = np.zeros((0, 5), dtype=np.float32)
        labels[i] = remain_labels

    return labels


def img2tensor(img):
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)  # ascontiguousarray
    img = torch.from_numpy(img).unsqueeze(0)
    return img


def img2label_paths(img_files):
    label_files = [
        x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
        for x in img_files
    ]
    return label_files


def load_img_files_from_txt(path):
    try:
        img_files = []  # image files
        for p in path if isinstance(path, list) else [path]:
            p = str(Path(p))  # os-agnostic
            parent = str(Path(p).parent) + os.sep
            if os.path.isfile(p):  # file
                with open(p, 'r') as t:
                    t = t.read().splitlines()
                    img_files += [
                        x.replace('./', parent) if x.startswith('./') else x
                        for x in t
                    ]  # local to global path
            elif os.path.isdir(p):  # folder
                img_files += glob.iglob(p + os.sep + '*.*')
            else:
                raise Exception('%s does not exist' % p)
        img_files = sorted([
            x.replace('/', os.sep) for x in img_files
            if os.path.splitext(x)[-1].lower() in img_formats
        ])
        return img_files

    except Exception as e:
        raise Exception('Error loading data from %s: %s\n' % (path, e))


# load_image
def load_image(path, resize_wh=None):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    if resize_wh is not None:
        # cv2.resize: w, h
        img = cv2.resize(img, (resize_wh[0], resize_wh[1]),
                         interpolation=cv2.INTER_LINEAR)
    return img, (h0, w0)  # im, hw_original


